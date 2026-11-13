"""
Batch推理: 1次I/O → N次计算。batch=4, 验证token/s提升。
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
from concurrent.futures import ThreadPoolExecutor

DEV="cuda";MODEL_DIR=r'D:\ai\models\qwen3.5-397b-bf16';NT=12;B=4
sys.path.insert(0,r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL_DIR,trust_remote_code=True).text_config
st=NonExpertStreamer(MODEL_DIR);rd=SafeExpertReader(MODEL_DIR)
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types
NV=64;NK=16;HK=128;HV=128;EP=cf.rms_norm_eps
pool=ThreadPoolExecutor(NT)

# Embed
with open(MODEL_DIR+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MODEL_DIR+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
m=hdr[ek];s,e=m['data_offsets'];fd=os.open(MODEL_DIR+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET);emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV);os.close(fd)

# RAM: CPU tensors
print('Loading layers...',end='',flush=True);t0=time.perf_counter()
layer_cpu=[{} for _ in range(60)]
for lid in range(60):
    ne=st.read_layer(lid)
    for k,v in ne.items():
        if 'mlp.experts.gate_up' in k or 'mlp.experts.down_proj' in k: continue
        if v.dtype==np.float32: layer_cpu[lid][k]=torch.from_numpy(v.copy())
        else: layer_cpu[lid][k]=torch.from_numpy(v.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16)
print(f' {time.perf_counter()-t0:.0f}s')

# KV
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

def attn(h,gpu,lid,bt,pre,cache):
    BS,SL,_=h.shape; pf=f'model.language_model.layers.{lid}.'
    def w(k): v=gpu.get(pf+k); return v.float() if v is not None else None
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        if q is None: return h*0.99
        Q=h.float()@q.T;K=h.float()@k.T;V=h.float()@v.T
        qa=Q[:,:,:8192].view(BS,SL,32,256).transpose(1,2)
        qg=Q[:,:,8192:].view(BS,SL,32,256).transpose(1,2)
        K=K.view(BS,SL,2,256).transpose(1,2).repeat_interleave(16,1)
        V=V.view(BS,SL,2,256).transpose(1,2).repeat_interleave(16,1)
        if not pre: kc,vc=cache.gkv(lid);K=torch.cat([kc,K],2) if kc is not None else K;V=torch.cat([vc,V],2) if vc is not None else V
        cache.ukv(lid,K,V)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
        return ((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(BS,SL,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight')
        a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight')
        o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight')
        dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None: return h*0.99
        KD=NK*HK;VD=NV*HV
        m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float()
            if not pre: cs=cache.gcv(lid);m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:SL];cache.ucv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(BS,SL,NK,HK);K=m[:,:,KD:2*KD].reshape(BS,SL,NK,HK);V=m[:,:,2*KD:].reshape(BS,SL,NV,HV)
        beta=(h.float()@b.T).sigmoid() if b is not None else torch.ones(BS,SL,NK,device=DEV)
        av=(h.float()@a.T) if a is not None else torch.zeros(BS,SL,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(BS,SL,NV,HK);K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(BS,SL,NV,HK)
        qn=l2norm(Q,-1,1e-6);kn=l2norm(K,-1,1e-6)
        init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.urc(lid,ls)
        at=ca.float().transpose(1,2).contiguous().view(BS,SL,VD)
        if nr is not None: ni=at.reshape(BS,SL,NV,HV);at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(BS,SL,VD)
        zz=(h.float()@z.T) if z is not None else at;at=at*F.silu(zz)
        return ((at@o.T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

# ═══════════ BATCH INFERENCE ═══════════
prompts=['你好','今天天气','Python','1+1=']
tokens_list=[tk.encode(p) for p in prompts]
SL=max(len(t) for t in tokens_list)
pad_id=tk.pad_token_id or 0
ids=torch.tensor([t+[pad_id]*(SL-len(t)) for t in tokens_list],device=DEV)
h=F.embedding(ids,emb)
cache=KVC()

print(f'Batch={B}, seq_len={SL}')
t0=time.perf_counter()
for lid in range(60):
    bt=LV[lid];pf=f'model.language_model.layers.{lid}.'
    gpu={k:v.to(DEV) for k,v in layer_cpu[lid].items()}

    inn=gpu.get(pf+'input_layernorm.weight')
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
    if inn is not None: hn=hn*inn.float()
    a_out=attn(hn.to(torch.bfloat16),gpu,lid,bt,True,cache)
    h=(h+a_out).to(torch.bfloat16)

    pn=gpu.get(pf+'post_attention_layernorm.weight')
    hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
    if pn is not None: hn2=hn2*pn.float()
    flat=hn2.to(torch.bfloat16).reshape(B*SL,H)
    rw=gpu.get(pf+'mlp.gate.weight')
    probs=torch.softmax(flat.float()@rw.float().T,-1)
    tw,ti=torch.topk(probs,10,-1)
    uids=ti.unique().tolist()[:10]

    futures={eid:pool.submit(rd.read_expert,lid,eid) for eid in uids}
    experts={}
    for eid,fut in futures.items():
        gn,dn=fut.result()
        experts[eid]=(torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16).float().to(DEV),
                     torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16).float().to(DEV))

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
    h=(h+(out+gv*so).to(torch.bfloat16).reshape(B,SL,H)).to(torch.bfloat16)

    del gpu,experts
    if lid%20==0: torch.cuda.empty_cache()

elapsed=time.perf_counter()-t0
logits=h[:,-1:,:].float()@emb.float().T/0.8
tokens=logits.argmax(-1)
print(f'Prefill: {elapsed:.0f}s ({B*SL/elapsed:.1f} tokens/s, {B/elapsed:.1f} prompts/s)')
for i,p in enumerate(prompts):
    tok=tk.decode([tokens[i].item()])
    print(f'  {p!r} -> {tok!r}')
print(f'vs single: 22s → {B}x batch = {B/elapsed*22:.1f}x throughput')
st.close();rd.close();pool.shutdown()
print(f'VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
