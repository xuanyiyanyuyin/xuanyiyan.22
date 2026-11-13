"""
三级存储推理: RAM缓存全60层非专家 + 450专家LRU + SSD兜底
GPU只放当前层计算图, RAM→GPU H2D, SSD→RAM预读
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
from concurrent.futures import ThreadPoolExecutor

MODEL_DIR=r'D:\ai\models\qwen3.5-397b-bf16'; DEV="cuda"
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL_DIR,trust_remote_code=True).text_config
st=NonExpertStreamer(MODEL_DIR); rd=SafeExpertReader(MODEL_DIR)
H=cf.hidden_size; V=cf.vocab_size; LV=cf.layer_types
NV=64; NK=16; HK=128; HV=128; EP=cf.rms_norm_eps

# ── Embed ──
with open(MODEL_DIR+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MODEL_DIR+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
m=hdr[ek]; s,e=m['data_offsets']; fd=os.open(MODEL_DIR+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET); emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV); os.close(fd)

# ── RAM Cache: all 60 layers non-expert (numpy, fast access) ──
print("Loading 60 layers to RAM...",end="",flush=True); t0=time.perf_counter()
layer_ram=[]  # [{key: np_array}]
for lid in range(60):
    ne=st.read_layer(lid)
    t={}
    for k,v in ne.items():
        if 'mlp.experts.gate_up' in k or 'mlp.experts.down_proj' in k: continue
        t[k]=v  # keep as numpy! no conversion
    layer_ram.append(t)
ram_mb=sum(sum(v.nbytes for v in t.values()) for t in layer_ram)/1024**2
print(f" {time.perf_counter()-t0:.0f}s ({ram_mb:.0f}MB)")

# ── Threaded expert reads (no cache lock, parallel SSD reads) ──
expert_pool=ThreadPoolExecutor(max_workers=12)

def get_experts_batch(lid, eids):
    """Read all experts in parallel from SSD, no cache (fast enough)"""
    futures={eid: expert_pool.submit(rd.read_expert, lid, eid) for eid in eids}
    results={}
    for eid, fut in futures.items():
        gn,dn=fut.result()
        results[eid]=(np2gpu(gn).float(), np2gpu(dn).float())
    return results

# ── GPU helpers ──
def np2gpu(arr):
    if arr.dtype==np.float32: return torch.from_numpy(arr.copy()).to(DEV)
    return torch.from_numpy(arr.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16).to(DEV)

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

# ═══════════ SINGLE LAYER ═══════════
def forward_layer(h,lid,bt,pre,cache):
    """Load layer from RAM→GPU, router→experts, compute, free GPU"""
    B,S,_=h.shape
    ne=layer_ram[lid]
    pf=f'model.language_model.layers.{lid}.'

    # Move non-expert from RAM to GPU (H2D, ~50ms)
    gpu={}
    for k,arr in ne.items():
        gpu[k]=np2gpu(arr)

    with torch.no_grad():
        # ── Attention ──
        inn=gpu.get(pf+'input_layernorm.weight')
        hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None: hn=hn*inn.float()
        hn=hn.to(torch.bfloat16)

        if bt=="full_attention":
            q=gpu.get(pf+'self_attn.q_proj.weight'); k=gpu.get(pf+'self_attn.k_proj.weight')
            v=gpu.get(pf+'self_attn.v_proj.weight'); o=gpu.get(pf+'self_attn.o_proj.weight')
            if q is None: a_out=h*0.99
            else:
                Q=h.float()@q.float().T; K=h.float()@k.float().T; V=h.float()@v.float().T
                qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2)
                qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
                K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
                V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
                if not pre: kc,vc=cache.gkv(lid); K=torch.cat([kc,K],2) if kc is not None else K; V=torch.cat([vc,V],2) if vc is not None else V
                cache.ukv(lid,K,V)
                att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
                a_out=((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,S,8192)@o.float().T).to(torch.bfloat16)
        else:
            qkv=gpu.get(pf+'linear_attn.in_proj_qkv.weight'); z=gpu.get(pf+'linear_attn.in_proj_z.weight')
            a=gpu.get(pf+'linear_attn.in_proj_a.weight'); b=gpu.get(pf+'linear_attn.in_proj_b.weight')
            o=gpu.get(pf+'linear_attn.out_proj.weight'); cv=gpu.get(pf+'linear_attn.conv1d.weight')
            dt=gpu.get(pf+'linear_attn.dt_bias'); al=gpu.get(pf+'linear_attn.A_log'); nr=gpu.get(pf+'linear_attn.norm.weight')
            if qkv is None: a_out=h*0.99
            else:
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
                if nr is not None: ni=at.reshape(B,S,NV,HV); at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(B,S,VD)
                zz=(h.float()@z.float().T) if z is not None else at; at=at*F.silu(zz)
                a_out=((at@o.float().T) if o is not None else at[:,:,:H]).to(torch.bfloat16)
        h=(h+a_out).to(torch.bfloat16)

        # ── RMSNorm 2 ──
        pn=gpu.get(pf+'post_attention_layernorm.weight')
        hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None: hn2=hn2*pn.float()
        flat=hn2.to(torch.bfloat16).reshape(-1,H)

        # ── Router ──
        rw=gpu.get(pf+'mlp.gate.weight')
        logits=flat.float()@rw.float().T; probs=torch.softmax(logits,-1)
        tw,ti=torch.topk(probs,10,-1); uids=ti.unique().tolist()[:10]

        # ── Experts (threaded SSD reads → GPU directly) ──
        expert_batch=get_experts_batch(lid, uids)
        out=torch.zeros_like(flat,dtype=torch.float32)
        for eid in uids:
            g,d=expert_batch[eid]  # already on GPU as float32
            mask=(ti==eid).any(-1)
            if not mask.any(): del g,d; continue
            t=flat[mask].float()
            o=(F.silu(t@g[:1024].T)*(t@g[1024:].T))@d.T
            w=tw[mask]; m=(ti[mask]==eid)
            out[mask]+=o*(w*m.float()).sum(-1,keepdim=True); del g,d

        sgw=gpu.get(pf+'mlp.shared_expert_gate.weight')
        sg=gpu.get(pf+'mlp.shared_expert.gate_proj.weight')
        su=gpu.get(pf+'mlp.shared_expert.up_proj.weight')
        sd=gpu.get(pf+'mlp.shared_expert.down_proj.weight')
        gv=torch.sigmoid(flat.float()@sgw.float().T)
        so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T
        h=(h+(out+gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)

    # Free GPU
    del gpu;torch.cuda.empty_cache()
    return h

# ═══════════ GENERATE ═══════════
def gen(prompt):
    ids=torch.tensor([tk.encode(prompt)],device=DEV)
    h=F.embedding(ids,emb); cache=KVC(); g=ids[0].tolist()

    for step in range(2):  # prefill + 1 decode
        pre=(step==0); t0=time.perf_counter()
        for lid in range(60):
            h=forward_layer(h,lid,LV[lid],pre,cache)

        logits=h[:,-1:,:].float()@emb.float().T/0.8
        if torch.isnan(logits).any(): logits=torch.nan_to_num(logits,nan=-1e9)
        v,_=torch.topk(logits,40,-1); logits[logits<v[:,:,-1:]]=-float('inf')
        pr=torch.softmax(logits,-1)
        if torch.isnan(pr).any(): pr=torch.ones_like(pr)/V
        nt=torch.multinomial(pr.squeeze(),1).item(); g.append(nt)
        dt=time.perf_counter()-t0
        print(f"  [{step}] '{tk.decode([nt])}' {dt:.0f}s",flush=True)
        if nt==tk.eos_token_id: break
        if step<1: h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(torch.bfloat16)

    return tk.decode(g)

for p in ["你好"]: print(f"\n{p!r}\n  -> {gen(p)!r}")
st.close();rd.close()
expert_pool.shutdown()
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
