"""
Qwen3.5-397B 推理 — GPU权重缓存 + pinned H2D
跨token复用非专家权重, 仅重新加载专家
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct, threading, queue

MODEL_DIR=r'D:\ai\models\qwen3.5-397b-bf16'; DEV="cuda"
sys.path.insert(0,r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from gpu_cache import GPUCache
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL_DIR,trust_remote_code=True).text_config
st=NonExpertStreamer(MODEL_DIR); rd=SafeExpertReader(MODEL_DIR)
H=cf.hidden_size; V=cf.vocab_size; LV=cf.layer_types
NV=64; NK=16; HK=128; HV=128; EP=cf.rms_norm_eps

# ── Embed (GPU resident) ──
with open(MODEL_DIR+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MODEL_DIR+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
m=hdr[ek]; s,e=m['data_offsets']; fd=os.open(MODEL_DIR+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET); emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16)).view(torch.bfloat16).reshape(V,H).float().to(DEV); os.close(fd)

# ── Layer numpy cache + GPU cache ──
layer_np = [st.read_layer(l) for l in range(60)]
gpu_cache = GPUCache(max_bytes=6*1024**3)  # 6GB for non-expert weights

def load_weight(key, ne, pf):
    """Load ONE weight from numpy → torch CPU (lazy, only called on cache miss)"""
    full = pf+key
    v = ne.get(full)
    if v is None: return None
    if v.dtype == np.float32: return torch.from_numpy(v)
    return torch.from_numpy(v.view(np.uint16).astype(np.int16)).view(torch.bfloat16)

# ── Threaded expert prefetch ──
expert_ram={}; eq=queue.Queue()
def prefetch_worker():
    while True:
        item=eq.get()
        if item is None: break
        lid,eids=item
        for eid in eids:
            if (lid,eid) not in expert_ram:
                gn,dn=rd.read_expert(lid,eid)
                expert_ram[(lid,eid)]=(torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16),
                                      torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16))
NT=12
for _ in range(NT): threading.Thread(target=prefetch_worker,daemon=True).start()

# ── KV Cache ──
class KVC:
    def __init__(self): self.kv={}; self.cv={}; self.rc={}
    def gkv(self,l): return self.kv.get(l)
    def gcv(self,l): return self.cv.get(l)
    def grc(self,l): return self.rc.get(l)
    def ukv(self,l,k,v):
        if l in self.kv: self.kv[l]=(torch.cat([self.kv[l][0],k],2),torch.cat([self.kv[l][1],v],2))
        else: self.kv[l]=(k,v)
    def ucv(self,l,s): self.cv[l]=s
    def urc(self,l,s): self.rc[l]=s

# ── Attention (uses gpu_cache) ──
def attn(h,ne,lid,bt,pre,cache):
    B,S,_=h.shape; pf=f'model.language_model.layers.{lid}.'
    def w(k): return gpu_cache.get(f'{lid}/{k}', lambda: load_weight(k,ne,pf))
    if bt=="full_attention":
        q=w('self_attn.q_proj.weight'); k=w('self_attn.k_proj.weight')
        v=w('self_attn.v_proj.weight'); o=w('self_attn.o_proj.weight')
        if q is None: return h*0.99
        Q=h.float()@q.float().T; K=h.float()@k.float().T; V=h.float()@v.float().T
        qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2); qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
        K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1); V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        if not pre: kc,vc=cache.gkv(lid); K=torch.cat([kc,K],2) if kc is not None else K; V=torch.cat([vc,V],2) if vc is not None else V
        cache.ukv(lid,K,V)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
        return ((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,S,8192)@o.float().T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight'); z=w('linear_attn.in_proj_z.weight')
        a=w('linear_attn.in_proj_a.weight'); b=w('linear_attn.in_proj_b.weight')
        o=w('linear_attn.out_proj.weight'); cv=w('linear_attn.conv1d.weight')
        dt=w('linear_attn.dt_bias'); al=w('linear_attn.A_log'); nr=w('linear_attn.norm.weight')
        if qkv is None: return h*0.99
        KD=NK*HK; VD=NV*HV
        m=h.float()@qkv.float().T; m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float()
            if not pre: cs=cache.gcv(lid); m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]; cache.ucv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK); K=m[:,:,KD:2*KD].reshape(B,S,NK,HK); V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.float().T).sigmoid() if b is not None else torch.ones(B,S,NK,device=DEV)
        av=(h.float()@a.float().T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK; Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK); K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
        qn=l2norm(Q,-1,1e-6); kn=l2norm(K,-1,1e-6)
        init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.urc(lid,ls)
        at=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nr is not None: ni=at.reshape(B,S,NV,HV); at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.float().view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h.float()@z.float().T) if z is not None else at; at=at*F.silu(zz)
        return ((at@o.float().T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

# ── MoE ──
def moe(flat,rw,lid,sgw,sg,su,sd):
    logits=flat.float()@rw.float().T; probs=torch.softmax(logits,-1)
    _,ti=torch.topk(probs,10,-1); uids=ti.unique().tolist()[:10]
    eq.put((lid, list(range(20))))  # prefetch next 20 experts

    out=torch.zeros_like(flat,dtype=torch.float32)
    for eid in uids:
        if (lid,eid) not in expert_ram:
            gn,dn=rd.read_expert(lid,eid)
            expert_ram[(lid,eid)]=(torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16),
                                   torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16))
        gc,dc=expert_ram[(lid,eid)]; g=gc.float().to(DEV); d=dc.float().to(DEV)
        mask=(ti==eid).any(-1)
        if not mask.any(): del g,d; continue
        t=flat[mask].float()
        o=(F.silu(t@g[:1024].T)*(t@g[1024:].T))@d.T
        tw=probs[mask]; eh=(ti[mask]==eid)
        out[mask]+=o*(tw*eh).sum(-1,keepdim=True); del g,d

    gv=torch.sigmoid(flat.float()@sgw.float().T)
    so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T
    return (out+gv*so).to(torch.bfloat16)

# ── Generate ──
def gen(prompt):
    ids=torch.tensor([tk.encode(prompt)],device=DEV)
    h=F.embedding(ids,emb); cache=KVC(); g=ids[0].tolist()

    for step in range(5):
        pre=(step==0); t0=time.perf_counter()
        for lid in range(60):
            ne=layer_np[lid]; bt=LV[lid]; pf=f'model.language_model.layers.{lid}.'
            w=lambda k: gpu_cache.get(f'{lid}/{k}', lambda kk=k: load_weight(kk,ne,pf))

            inn=w('input_layernorm.weight'); pn=w('post_attention_layernorm.weight')
            sgw=w('mlp.shared_expert_gate.weight'); sg=w('mlp.shared_expert.gate_proj.weight')
            su=w('mlp.shared_expert.up_proj.weight'); sd=w('mlp.shared_expert.down_proj.weight')
            rw=w('mlp.gate.weight')

            with torch.no_grad():
                hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
                if inn is not None: hn=hn*inn.float()
                a_out=attn(hn.to(torch.bfloat16),ne,lid,bt,pre,cache)
                h=(h+a_out).to(torch.bfloat16)

                hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
                if pn is not None: hn2=hn2*pn.float()
                moe_out=moe(hn2.to(torch.bfloat16).reshape(-1,H),rw,lid,sgw,sg,su,sd)
                h=(h+moe_out.reshape_as(h)).to(torch.bfloat16)

        logits=h[:,-1:,:].float()@emb.float().T/0.8
        if torch.isnan(logits).any(): logits=torch.nan_to_num(logits,nan=-1e9)
        v,_=torch.topk(logits,40,-1); logits[logits<v[:,:,-1:]]=-float('inf')
        pr=torch.softmax(logits,-1)
        if torch.isnan(pr).any(): pr=torch.ones_like(pr)/V
        nt=torch.multinomial(pr.squeeze(),1).item(); g.append(nt)
        print(f"  [{step}] '{tk.decode([nt])}' ({time.perf_counter()-t0:.0f}s) cache: {gpu_cache.stats()}",flush=True)
        if nt==tk.eos_token_id: break
        if step<4: h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(torch.bfloat16)

    return tk.decode(g)

for p in ["你好"]: print(f"\n{p!r}\n  -> {gen(p)!r}")
for _ in range(NT): eq.put(None)
st.close(); rd.close()
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
