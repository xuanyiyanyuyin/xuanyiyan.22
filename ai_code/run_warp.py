"""
GPU常驻小权重(norms+shared+router=1.2GB) + 注意力按需加载 + 专家读与计算重叠
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
from concurrent.futures import ThreadPoolExecutor

DEV="cuda"; MODEL_DIR=r'D:\ai\models\qwen3.5-397b-bf16'; NT=12
sys.path.insert(0,r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL_DIR,trust_remote_code=True).text_config
st=NonExpertStreamer(MODEL_DIR); rd=SafeExpertReader(MODEL_DIR)
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types
NV=64;NK=16;HK=128;HV=128;EP=cf.rms_norm_eps
pool=ThreadPoolExecutor(NT)

# Embed
with open(MODEL_DIR+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MODEL_DIR+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
m=hdr[ek];s,e=m['data_offsets'];fd=os.open(MODEL_DIR+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET);emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV);os.close(fd)

# ── Preload: split weights into RESIDENT (GPU) and ON-DEMAND (CPU) ──
RESIDENT_KEYS = {'input_layernorm.weight', 'post_attention_layernorm.weight',
    'mlp.shared_expert_gate.weight', 'mlp.shared_expert.gate_proj.weight',
    'mlp.shared_expert.up_proj.weight', 'mlp.shared_expert.down_proj.weight',
    'mlp.gate.weight', 'linear_attn.norm.weight', 'self_attn.q_norm.weight',
    'self_attn.k_norm.weight', 'linear_attn.dt_bias', 'linear_attn.A_log',
    'linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight'}

print('Loading to RAM + GPU resident...',end='',flush=True);t0=time.perf_counter()
gpu_resident=[{} for _ in range(60)]  # small weights, stay on GPU
cpu_ondemand=[{} for _ in range(60)]  # attention weights, CPU tensor

for lid in range(60):
    ne=st.read_layer(lid); pf=f'model.language_model.layers.{lid}.'
    for k,v in ne.items():
        t=None
        if v.dtype==np.float32: t=torch.from_numpy(v.copy())
        else: t=torch.from_numpy(v.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16)
        short_key=k[len(pf):]
        if short_key in RESIDENT_KEYS or any(rk in short_key for rk in RESIDENT_KEYS):
            gpu_resident[lid][k]=t.to(DEV)
        elif 'mlp.experts.' not in k:
            cpu_ondemand[lid][k]=t  # stay on CPU

gpu_mb=sum(sum(v.numel()*v.element_size() for v in d.values()) for d in gpu_resident)/1024**2
cpu_mb=sum(sum(v.numel()*v.element_size() for v in d.values()) for d in cpu_ondemand)/1024**2
print(f' {time.perf_counter()-t0:.0f}s (GPU:{gpu_mb:.0f}MB CPU:{cpu_mb:.0f}MB)')

# KV Cache
class KVC:
    def __init__(self): self.kv={};self.cv={};self.rc={}
    def gkv(self,l): return self.kv.get(l)
    def gcv(self,l): return self.cv.get(l)
    def grc(self,l): return self.rc.get(l)
    def ukv(self,l,k,v):
        if l in self.kv: self.kv[l]=(torch.cat([self.kv[l][0],k],2),torch.cat([self.kv[l][1],v],2))
        else: self.kv[l]=(k,v)
    def ucv(self,l,s): self.cv[l]=s
    def urc(self,l,s): self.rc[l]=s

# Attention
def attn(h,g,lid,bt,pre,cache):
    B,S,_=h.shape;pf=f'model.language_model.layers.{lid}.'
    def w(k): v=g.get(pf+k); return v.float() if v is not None else None
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        if q is None: return h*0.99
        Q=h.float()@q.T;K=h.float()@k.T;V=h.float()@v.T
        qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2);qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
        K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1);V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        if not pre: kc,vc=cache.gkv(lid);K=torch.cat([kc,K],2) if kc is not None else K;V=torch.cat([vc,V],2) if vc is not None else V
        cache.ukv(lid,K,V)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
        return ((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,S,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight')
        o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');nr=w('linear_attn.norm.weight')
        a=gpu_resident[lid].get(pf+'linear_attn.in_proj_a.weight')
        b=gpu_resident[lid].get(pf+'linear_attn.in_proj_b.weight')
        dt=gpu_resident[lid].get(pf+'linear_attn.dt_bias')
        al=gpu_resident[lid].get(pf+'linear_attn.A_log')
        if qkv is None: return h*0.99
        KD=NK*HK;VD=NV*HV
        m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float()
            if not pre: cs=cache.gcv(lid);m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S];cache.ucv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.float().T).sigmoid() if b is not None else torch.ones(B,S,NK,device=DEV)
        av=(h.float()@a.float().T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK);K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
        qn=l2norm(Q,-1,1e-6);kn=l2norm(K,-1,1e-6)
        init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.urc(lid,ls)
        at=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nr is not None: ni=at.reshape(B,S,NV,HV);at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h.float()@z.float().T) if z is not None else at;at=at*F.silu(zz)
        return ((at@o.T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

# ═══════════ FORWARD ═══════════
print('='*50)
print('GPU RESIDENT + overlaid expert reads')
print('='*50)

ids=torch.tensor([tk.encode('你好')],device=DEV)
h=F.embedding(ids,emb);cache=KVC()

for step in range(2):
    pre=(step==0);t0=time.perf_counter()
    for lid in range(60):
        bt=LV[lid];pf=f'model.language_model.layers.{lid}.'

        # H2D attention weights only (from CPU tensors)
        gpu={}
        for k,v in cpu_ondemand[lid].items():
            gpu[k]=v.to(DEV,non_blocking=True)
        # Merge GPU-resident weights
        for k,v in gpu_resident[lid].items():
            gpu[k]=v

        # Attention
        inn=gpu.get(pf+'input_layernorm.weight')
        hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None: hn=hn*inn.float()
        a_out=attn(hn.to(torch.bfloat16),gpu,lid,bt,pre,cache)
        h=(h+a_out).to(torch.bfloat16)

        # Router
        pn=gpu.get(pf+'post_attention_layernorm.weight')
        hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None: hn2=hn2*pn.float()
        flat=hn2.to(torch.bfloat16).reshape(-1,H)
        rw=gpu.get(pf+'mlp.gate.weight')
        probs=torch.softmax(flat.float()@rw.float().T,-1)
        tw,ti=torch.topk(probs,10,-1);uids=ti.unique().tolist()[:10]

        # Threaded expert reads
        futures={eid:pool.submit(rd.read_expert,lid,eid) for eid in uids}
        experts={}
        for eid,fut in futures.items():
            gn,dn=fut.result()
            experts[eid]=(torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16).float().to(DEV),
                         torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16).float().to(DEV))

        # MoE
        sgw=gpu.get(pf+'mlp.shared_expert_gate.weight');sg=gpu.get(pf+'mlp.shared_expert.gate_proj.weight')
        su=gpu.get(pf+'mlp.shared_expert.up_proj.weight');sd=gpu.get(pf+'mlp.shared_expert.down_proj.weight')
        out=torch.zeros_like(flat,dtype=torch.float32)
        for eid in uids:
            g,d=experts[eid]
            mask=(ti==eid).any(-1)
            if not mask.any(): continue
            t2=flat[mask].float()
            out[mask]+=(F.silu(t2@g[:1024].T)*(t2@g[1024:].T))@d.T
        gv=torch.sigmoid(flat.float()@sgw.float().T)
        so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T
        h=(h+(out+gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)

        del gpu,experts
        if lid%30==0: torch.cuda.empty_cache()

    elapsed=time.perf_counter()-t0
    print(f"  [{step}] {elapsed:.0f}s",flush=True)
    logits=h[:,-1:,:].float()@emb.float().T/0.8
    if torch.isnan(logits).any(): logits=torch.nan_to_num(logits,nan=-1e9)
    v,_=torch.topk(logits,40,-1);logits[logits<v[:,:,-1:]]=-float('inf')
    pr=torch.softmax(logits,-1)
    if torch.isnan(pr).any(): pr=torch.ones_like(pr)/V
    nt=torch.multinomial(pr.squeeze(),1).item()
    if step<1: h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(torch.bfloat16)

st.close();rd.close();pool.shutdown()
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
