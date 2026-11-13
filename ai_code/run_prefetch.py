"""
专家预读流水线: 当前层路由决定下一层预读 + 投机预读下下层
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
from concurrent.futures import ThreadPoolExecutor

DEV="cuda";MD=r'D:\ai\models\qwen3.5-397b-bf16';NT=12
sys.path.insert(0,r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MD,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MD,trust_remote_code=True).text_config
st=NonExpertStreamer(MD);rd=SafeExpertReader(MD)
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types
NV=64;NK=16;HK=128;HV=128;EP=cf.rms_norm_eps
pool=ThreadPoolExecutor(NT)

# Embed
with open(MD+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MD+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(n))
m=hdr[ek];s,e=m['data_offsets'];fd=os.open(MD+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET);emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV);os.close(fd)

# RAM cache (CPU tensors)
print('Load RAM...',end='',flush=True);t0=time.perf_counter()
layer_cpu=[]
for lid in range(60):
    ne=st.read_layer(lid);t={}
    for k,v in ne.items():
        if 'mlp.experts.gate_up' in k or 'mlp.experts.down_proj' in k: continue
        if v.dtype==np.float32: t[k]=torch.from_numpy(v.copy())
        else: t[k]=torch.from_numpy(v.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16)
    layer_cpu.append(t)
print(f'{time.perf_counter()-t0:.0f}s')

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
    B,S,_=h.shape;pf=f'model.language_model.layers.{lid}.'
    def w(k): v=gpu.get(pf+k); return v.float() if v is not None else None
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
        a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight')
        o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight')
        dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None: return h*0.99
        KD=NK*HK;VD=NV*HV
        m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float()
            if not pre: cs=cache.gcv(lid);m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S];cache.ucv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.T).sigmoid() if b is not None else torch.ones(B,S,NK,device=DEV)
        av=(h.float()@a.T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK);K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
        qn=l2norm(Q,-1,1e-6);kn=l2norm(K,-1,1e-6)
        init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.urc(lid,ls)
        at=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nr is not None: ni=at.reshape(B,S,NV,HV);at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h.float()@z.T) if z is not None else at;at=at*F.silu(zz)
        return ((at@o.T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

# ═══════════ PREFETCH SYSTEM ═══════════
prefetched = {}  # lid → {eid: (gate_up_gpu, down_gpu)}
prefetch_futures = {}  # lid → Future
hits = 0; misses = 0

def launch_prefetch(lid, eids):
    """Fire background read for specified experts at layer lid"""
    key = lid
    if key in prefetch_futures: return  # already prefetching
    def do_read():
        results = {}
        for eid in eids:
            gn, dn = rd.read_expert(lid, eid)
            # Store as CPU tensor, NOT GPU! GPU conversion happens on main thread
            results[eid] = (torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16),
                           torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16))
        prefetched[lid] = results
    prefetch_futures[key] = pool.submit(do_read)

def get_experts(lid, eids):
    """Get experts: prefetched (CPU)→GPU or sync read"""
    global hits, misses
    if lid in prefetched:
        cpu_cache = prefetched.pop(lid)
        missing = [e for e in eids if e not in cpu_cache]
        if not missing:
            hits += 1
            # Batch H2D: stack all experts → one GPU transfer (240MB at once)
            gus=torch.stack([cpu_cache[e][0] for e in eids]).float().to(DEV)
            dns=torch.stack([cpu_cache[e][1] for e in eids]).float().to(DEV)
            return {e: (gus[i], dns[i]) for i,e in enumerate(eids)}
        misses += 1
        for eid in missing:
            gn, dn = rd.read_expert(lid, eid)
            cpu_cache[eid] = (torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16),
                            torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16))
        gus=torch.stack([cpu_cache[e][0] for e in eids]).float().to(DEV)
        dns=torch.stack([cpu_cache[e][1] for e in eids]).float().to(DEV)
        return {e: (gus[i], dns[i]) for i,e in enumerate(eids)}
    else:
        misses += 1
        results = {}
        gn_list=[]; dn_list=[]
        for eid in eids:
            gn, dn = rd.read_expert(lid, eid)
            gn_list.append(torch.from_numpy(gn.astype(np.int16)).view(torch.bfloat16))
            dn_list.append(torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16))
        gu=torch.stack(gn_list).float().to(DEV); dw=torch.stack(dn_list).float().to(DEV)
        return {e: (gu[i], dw[i]) for i,e in enumerate(eids)}

# ═══════════ FORWARD ═══════════
ids=torch.tensor([tk.encode('你好')],device=DEV)
h=F.embedding(ids,emb);cache=KVC()
prev_routing={}  # lid → list of eids

T={'attn':0,'exp':0,'moe':0};T_hit=0;T_miss=0

print(f'\n{"L":>3s} {"T":>4s} {"attn":>7s} {"exp":>7s} {"moe":>7s} {"tot":>6s} {"st":>4s}')
print('-'*50)

for lid in range(60):
    bt=LV[lid];pf=f'model.language_model.layers.{lid}.'
    gpu={k:v.to(DEV) for k,v in layer_cpu[lid].items()}

    t0=time.perf_counter()

    inn=gpu.get(pf+'input_layernorm.weight')
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
    if inn is not None: hn=hn*inn.float()
    a_out=attn(hn.to(torch.bfloat16),gpu,lid,bt,True,cache)
    h=(h+a_out).to(torch.bfloat16)
    t1=time.perf_counter()

    pn=gpu.get(pf+'post_attention_layernorm.weight')
    hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
    if pn is not None: hn2=hn2*pn.float()
    flat=hn2.to(torch.bfloat16).reshape(-1,H)
    rw=gpu.get(pf+'mlp.gate.weight')
    probs=torch.softmax(flat.float()@rw.float().T,-1)
    tw,ti=torch.topk(probs,10,-1);uids=ti.unique().tolist()[:10]

    # ── Get experts (prefetched or sync) ──
    is_hit = lid in prefetched
    experts=get_experts(lid, uids)
    if is_hit: T_hit+=1
    t2=time.perf_counter()

    # ── Launch prefetch for LAYERS AHEAD ──
    # Predict: layer N+1 will use same experts as layer N (inertial)
    if lid+1 < 60:
        launch_prefetch(lid+1, uids)
    # Predict: layer N+2 also same
    if lid+2 < 60:
        launch_prefetch(lid+2, uids)

    sgw=gpu.get(pf+'mlp.shared_expert_gate.weight');sg=gpu.get(pf+'mlp.shared_expert.gate_proj.weight')
    su=gpu.get(pf+'mlp.shared_expert.up_proj.weight');sd=gpu.get(pf+'mlp.shared_expert.down_proj.weight')
    out=torch.zeros_like(flat,dtype=torch.float32)
    for eid in uids:
        g,d=experts[eid]
        mask=(ti==eid).any(-1)
        if not mask.any(): continue
        t3=flat[mask].float()
        out[mask]+=(F.silu(t3@g[:1024].T)*(t3@g[1024:].T))@d.T
    gv=torch.sigmoid(flat.float()@sgw.float().T)
    so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T
    h=(h+(out+gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)
    t_end=time.perf_counter()

    a=t1-t0;e=t2-t1;m=t_end-t2;tot=a+e+m
    T['attn']+=a;T['exp']+=e;T['moe']+=m
    status='HIT' if is_hit else 'MIS'
    tl='full' if bt=='full_attention' else 'lin'
    print(f'{lid:3d} {tl:>4s} {a*1000:6.0f}ms {e*1000:6.0f}ms {m*1000:6.0f}ms {tot*1000:5.0f}ms {status:>4s}')

    del gpu,experts
    if lid%20==0: torch.cuda.empty_cache()

total=sum(T.values())
print(f'\nTotal: {total:.1f}s (attn={T["attn"]:.1f}s exp={T["exp"]:.1f}s moe={T["moe"]:.1f}s)')
print(f'Prefetch: {T_hit} hits, {60-T_hit} misses ({T_hit/60*100:.0f}% hit rate)')
st.close();rd.close();pool.shutdown()
