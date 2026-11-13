"""
122B MoE vFINAL: safetensors全部非专家 + mmap专家
不用NonExpertStreamer, 不用numpy转换, 原始精度
"""
import torch, torch.nn.functional as F, time, json, os, sys
DEV='cuda';sys.path.insert(0,r'D:\ai')
from safetensors.torch import load_file as sf_load
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm
from expert_reader import SafeExpertReader

MODEL=r'D:\ai\models\Qwen3.5-122B-A10B'

tk=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL,trust_remote_code=True).text_config
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types;EP=cf.rms_norm_eps
NV,NK,HK,HV=64,16,128,128
print(f'122B: {cf.num_hidden_layers} layers, H={H}, V={V}')

# ═══ safetensors load ALL non-expert shards (skip experts) ═══
print('Loading non-expert...',end='',flush=True);t0=time.perf_counter()
idx=json.load(open(MODEL+'/model.safetensors.index.json'))
layer_ram=[{} for _ in range(48)]
emb=None; loaded=0;skipped=0
for shard in set(idx['weight_map'].values()):
    has_non=any('.mlp.experts.' not in n for n,s in idx['weight_map'].items() if s==shard)
    if not has_non:continue
    d=sf_load(MODEL+'/'+shard,device='cpu')
    for k,v in d.items():
        if '.mlp.experts.' in k: skipped+=1;continue
        loaded+=1;name=k.replace('model.language_model.','')
        if 'embed_tokens' in name:emb=v.float().to(DEV)
        if 'lm_head' in name:lm_head=v.float().to(DEV)
        elif 'layers.' in name:
            p=name.split('.');lid=int(p[p.index('layers')+1]);key='.'.join(p[p.index('layers')+2:])
            layer_ram[lid][key]=v
gb=sum(sum(vv.numel()*vv.element_size() for vv in d.values()) for d in layer_ram)/1024**3
print(f' {time.perf_counter()-t0:.0f}s ({gb:.1f}GB RAM, {loaded} non-exp, {skipped} exp)')

rd=SafeExpertReader(MODEL)

# ═══ KV Cache ═══
class KVC:
    def __init__(self):self.kv={};self.cv={};self.rc={}
    def gkv(self,l):return self.kv.get(l)
    def gcv(self,l):return self.cv.get(l)
    def grc(self,l):return self.rc.get(l)
    def ukv(self,l,k,v):
        if l in self.kv:
            nk=torch.cat([self.kv[l][0],k],2)[:,:,-128:]
            nv=torch.cat([self.kv[l][1],v],2)[:,:,-128:];self.kv[l]=(nk,nv)
        else:self.kv[l]=(k[:,:,-128:].clone() if k.shape[2]>128 else k,v[:,:,-128:].clone() if v.shape[2]>128 else v)
    def ucv(self,l,s):self.cv[l]=s[:,:,-4:].clone() if s.shape[2]>4 else s
    def urc(self,l,s):self.rc[l]=s

# ═══ Attention ═══
def attn(h,gpu,lid,bt,pre,cache):
    B,S,_=h.shape
    def w(k):v=gpu.get(k);return v.float().to(DEV) if v is not None else None
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        qn=gpu.get('self_attn.q_norm.weight');kn=gpu.get('self_attn.k_norm.weight')
        if q is None:return h*0.99
        Q=h.float()@q.T;K=h.float()@k.T;V=h.float()@v.T
        qa=Q[:,:,:8192].view(B,S,32,256);qg=Q[:,:,8192:].view(B,S,32,256)
        if qn is not None:
            qa=qa*torch.rsqrt(qa.float().pow(2).mean(-1,keepdim=True)+EP)*(1.0+qn.float().to(DEV))
            qg=qg*torch.rsqrt(qg.float().pow(2).mean(-1,keepdim=True)+EP)*(1.0+qn.float().to(DEV))
        qa=qa.transpose(1,2);qg=qg.transpose(1,2)
        K=K.view(B,S,2,256)
        if kn is not None:K=K*torch.rsqrt(K.float().pow(2).mean(-1,keepdim=True)+EP)*(1.0+kn.float().to(DEV))
        K=K.transpose(1,2).repeat_interleave(16,1);V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        # RoPE: rotate first rope_dim channels
        Hq, D = qa.shape[1], qa.shape[3]
        pids = torch.arange(S, device=DEV) if pre else torch.tensor([step_cnt], device=DEV)
        c = rope_cos[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        s = rope_sin[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        x_rot = qa[:,:,:,:rope_dim]
        x1, x2 = x_rot[..., :rope_dim//2], x_rot[..., rope_dim//2:]
        c1, c2 = c[..., :rope_dim//2], c[..., rope_dim//2:]
        s1, s2 = s[..., :rope_dim//2], s[..., rope_dim//2:]
        rotated = torch.cat([x1*c1 - x2*s1, x2*c2 + x1*s2], dim=-1)
        qa = torch.cat([rotated, qa[:,:,:,rope_dim:]], dim=-1)
        # Also apply RoPE to K before caching
        Kr = K[:,:,:,:rope_dim]
        K1, K2 = Kr[..., :rope_dim//2], Kr[..., rope_dim//2:]
        K_rot = torch.cat([K1*c1 - K2*s1, K2*c2 + K1*s2], dim=-1)
        K = torch.cat([K_rot, K[:,:,:,rope_dim:]], dim=-1)
        if not pre:
            kv=cache.gkv(lid)
            if kv is not None:K=torch.cat([kv[0],K],2);V=torch.cat([kv[1],V],2)
        cache.ukv(lid,K,V)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
        gated=torch.sigmoid(qg)*att
        if lid<5:print(f'L{lid} gate_sig={torch.sigmoid(qg).mean():.3f} att_n={att.norm():.1f} gated_n={gated.norm():.1f}',flush=True)
        return (gated.transpose(1,2).contiguous().view(B,S,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight');a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight');o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None:return h*0.99
        KD=NK*HK;VD=NV*HV;m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float()
            if not pre:
                cs=cache.gcv(lid)
                if cs is not None:m=torch.cat([cs,m],-1)
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S];cache.ucv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.T).sigmoid() if b is not None else torch.ones(B,S,NK,device=DEV)
        av=(h.float()@a.T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK);K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
        qn=l2norm(Q,-1,1e-6);kn=l2norm(K,-1,1e-6);init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True);cache.urc(lid,ls)
        at=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nr is not None:ni=at.reshape(B,S,NV,HV);at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*(1.0+nr.view(1,1,1,-1))).reshape(B,S,VD)
        zz=(h.float()@z.T) if z is not None else at;at=at*F.silu(zz)
        return ((at@o.T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

# ═══ INFERENCE ═══
PROMPT='请用中文介绍你自己'
print(f'\n"{PROMPT}"')
ids=tk.apply_chat_template([{'role':'user','content':PROMPT}],add_generation_prompt=True,return_tensors='pt',tokenize=True).input_ids.to(DEV)
h=F.embedding(ids,emb);cache=KVC()
# RoPE cos/sin cache
rope_dim = int(256 * 0.25)  # partial_rotary_factor
freq = 1.0/(10000000**(torch.arange(0,rope_dim,2,device=DEV).float()/rope_dim))
rope_cos = torch.zeros(512, rope_dim, device=DEV)
rope_sin = torch.zeros(512, rope_dim, device=DEV)
for p in range(512):
    theta = p * freq
    rope_cos[p] = torch.cat([theta.cos(), theta.cos()])
    rope_sin[p] = torch.cat([theta.sin(), theta.sin()])
t0=time.perf_counter();gen=[]
step_cnt=0
for step in range(32):
    pre=(step==0)
    for lid in range(48):
        bt=LV[lid];t=layer_ram[lid]
        gpu={k:v.float().to(DEV,non_blocking=True) for k,v in t.items()}
        inn=gpu.get('input_layernorm.weight');hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None:hn=hn*(1.0+inn.float())
        a_out=attn(hn.to(torch.bfloat16),gpu,lid,bt,pre,cache);h=(h+a_out).to(torch.bfloat16)
        pn=gpu.get('post_attention_layernorm.weight');hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None:hn2=hn2*(1.0+pn.float())
        flat=hn2.to(torch.bfloat16).reshape(-1,H);rw=gpu.get('mlp.gate.weight')
        probs=torch.softmax(flat.float()@rw.float().T,-1);tw,ti=torch.topk(probs,8,-1);tw=tw/tw.sum(-1,keepdim=True);uids=ti.unique().tolist()
        out=torch.zeros_like(flat,dtype=torch.float32)
        for eid in uids[:8]:
            try:gn,dn=rd.read_expert(lid,eid);g=gn.float().to(DEV);d=dn.float().to(DEV);mask=(ti==eid).any(-1)
            except:continue
            if mask.any():t2=flat[mask].float();eid_mask=(ti[mask]==eid);gw=(tw[mask]*eid_mask).sum(-1);out[mask]+=gw.unsqueeze(-1)*((F.silu(t2@g[:1024].T)*(t2@g[1024:].T))@d.T)
        sgw=gpu.get('mlp.shared_expert_gate.weight')
        if sgw is not None:
            sg=gpu.get('mlp.shared_expert.gate_proj.weight');su=gpu.get('mlp.shared_expert.up_proj.weight');sd=gpu.get('mlp.shared_expert.down_proj.weight')
            if sg is not None:
                gv=torch.sigmoid(flat.float()@sgw.float().T);so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T;out=out+gv*so
        if lid<5 or lid%10==0:
            rn=h.float().norm();mn=out.norm()
            print(f'L{lid:02d} res={rn:.1f} moe={mn:.2f} R={mn/rn:.3f}' if rn>0 else f'L{lid:02d} res=0',flush=True)
        h=(h+out.to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16);del gpu
        if lid%20==0:torch.cuda.empty_cache()
    logits=h[:,-1:,:].float()@lm_head.float().T/0.6;pr=torch.softmax(logits,-1);nt=pr.argmax(-1).item();gen.append(nt);step_cnt+=1
    print(f'[{step}] {tk.decode([nt])}',flush=True)
    if nt==tk.eos_token_id:break
    h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(torch.bfloat16)

print(f'\nOutput: {tk.decode(ids[0].tolist()+gen)[:300]}')
print(f'{time.perf_counter()-t0:.0f}s VRAM:{torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
rd.close()
