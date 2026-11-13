"""
组件级核对: 官方模块(随机权重) vs 手写实现, 相同输入比较输出
- Test 1: full_attention (Qwen3_5MoeAttention) prefill
- Test 2: linear_attention (GatedDeltaNet) prefill + decode(含状态衔接)
- Test 3: MoE block (router + experts + shared)
- Test 4: RMSNorm / RMSNormGated
- Test 5: RoPE 表
"""
import torch, torch.nn.functional as F, json
torch.manual_seed(42)
DEV='cuda'
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention, Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeRMSNorm, Qwen3_5MoeRMSNormGated, Qwen3_5MoeTextRotaryEmbedding,
    l2norm, torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
    torch_causal_conv1d_update,
)

MODEL=r'D:\ai\models\Qwen3.5-122B-A10B'
cfg=AutoConfig.from_pretrained(MODEL,trust_remote_code=True).text_config
cfg._attn_implementation='eager'
EP=cfg.rms_norm_eps
NV,NK,HK,HV=cfg.linear_num_value_heads,cfg.linear_num_key_heads,cfg.linear_key_head_dim,cfg.linear_value_head_dim
H=cfg.hidden_size
rope_dim=int(256*cfg.rope_parameters['partial_rotary_factor'])
print(f'cfg ok: H={H} rope_dim={rope_dim} NV={NV} NK={NK}')

def relerr(a,b):
    a=a.float();b=b.float()
    return ((a-b).abs().max().item(), (a-b).norm().item()/(b.norm().item()+1e-9))

# ═══ 手写实现的 RoPE 表 (与 test_122b_fixed.py 完全相同) ═══
MAXP=128
freq = 1.0/(cfg.rope_parameters['rope_theta']**(torch.arange(0,rope_dim,2,device=DEV).float()/rope_dim))
rope_cos = torch.zeros(MAXP, rope_dim, device=DEV)
rope_sin = torch.zeros(MAXP, rope_dim, device=DEV)
for p in range(MAXP):
    theta = p * freq
    rope_cos[p] = torch.cat([theta.cos(), theta.cos()])
    rope_sin[p] = torch.cat([theta.sin(), theta.sin()])

cur_pos=0

# ═══ 手写 attn (从 test_122b_fixed.py 原样复制) ═══
def attn(h,gpu,lid,bt,pre,cache,S_override=None):
    B,S,_=h.shape
    def w(k):v=gpu.get(k);return v.float().to(DEV) if v is not None else None
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        qn=gpu.get('self_attn.q_norm.weight');kn=gpu.get('self_attn.k_norm.weight')
        if q is None:return h*0.99
        Q=(h.float()@q.T).view(B,S,32,512)
        qa=Q[:,:,:,:256];qg=Q[:,:,:,256:]
        if qn is not None:
            qa=qa*torch.rsqrt(qa.pow(2).mean(-1,keepdim=True)+EP)*(1.0+qn.float().to(DEV))
        qa=qa.transpose(1,2);qg=qg.transpose(1,2)
        K=(h.float()@k.T).view(B,S,2,256)
        Vm=(h.float()@v.T).view(B,S,2,256)
        if kn is not None:K=K*torch.rsqrt(K.pow(2).mean(-1,keepdim=True)+EP)*(1.0+kn.float().to(DEV))
        K=K.transpose(1,2).repeat_interleave(16,1);Vm=Vm.transpose(1,2).repeat_interleave(16,1)
        Hq = qa.shape[1]
        pids = torch.arange(S, device=DEV) if pre else torch.tensor([cur_pos], device=DEV)
        c = rope_cos[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        s = rope_sin[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        x_rot = qa[:,:,:,:rope_dim]
        x1, x2 = x_rot[..., :rope_dim//2], x_rot[..., rope_dim//2:]
        c1, c2 = c[..., :rope_dim//2], c[..., rope_dim//2:]
        s1, s2 = s[..., :rope_dim//2], s[..., rope_dim//2:]
        rotated = torch.cat([x1*c1 - x2*s1, x2*c2 + x1*s2], dim=-1)
        qa = torch.cat([rotated, qa[:,:,:,rope_dim:]], dim=-1)
        Kr = K[:,:,:,:rope_dim]
        K1, K2 = Kr[..., :rope_dim//2], Kr[..., rope_dim//2:]
        K_rot = torch.cat([K1*c1 - K2*s1, K2*c2 + K1*s2], dim=-1)
        K = torch.cat([K_rot, K[:,:,:,rope_dim:]], dim=-1)
        if not pre:
            kv=cache.get('kv')
            if kv is not None:K=torch.cat([kv[0],K],2);Vm=torch.cat([kv[1],Vm],2)
        cache['kv']=(K,Vm)
        att=F.scaled_dot_product_attention(qa,K,Vm,is_causal=pre)
        gated=torch.sigmoid(qg)*att
        return (gated.transpose(1,2).contiguous().view(B,S,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight');a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight');o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None:return h*0.99
        KD=NK*HK;VD=NV*HV;m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float()
            if cw.dim()==2:cw=cw.unsqueeze(1)
            if not pre:
                cs=cache.get('conv')
                if cs is not None:
                    mcat=torch.cat([cs,m],-1)
                    cache['conv']=mcat[:,:,-3:].clone()
                    m=F.silu(F.conv1d(mcat,cw,padding=0,groups=12288))
                else:
                    cache['conv']=F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone()
                    m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
            else:
                cache['conv']=F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone()
                m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.T).sigmoid() if b is not None else torch.ones(B,S,NV,device=DEV)
        av=(h.float()@a.T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.repeat_interleave(R,2);K=K.repeat_interleave(R,2)
        init=None if pre else cache.get('rec')
        ca,ls=torch_chunk_gated_delta_rule(Q,K,V,g,beta,64,init,True,True);cache['rec']=ls
        at=ca.float().contiguous().view(B,S,VD)
        if nr is not None:
            ni=at.reshape(B,S,NV,HV)
            at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h.float()@z.T) if z is not None else at;at=at*F.silu(zz)
        return ((at@o.T) if o is not None else at[:,:,:H]).to(torch.bfloat16)

results=[]
def check(name, mine, ref, tol=2e-2):
    mx,re=relerr(mine,ref)
    ok = re < tol
    results.append((name,ok,mx,re))
    print(f'{"PASS" if ok else "FAIL"} {name}: max_abs={mx:.4f} rel={re:.5f}')

# ═══ Test 5: RoPE 表 ═══
rope_mod=Qwen3_5MoeTextRotaryEmbedding(cfg).to(DEV)
xdummy=torch.zeros(1,20,H,device=DEV,dtype=torch.bfloat16)
pids=torch.arange(20,device=DEV).view(1,1,-1).expand(3,1,-1)
cos_ref,sin_ref=rope_mod(xdummy,pids)  # (1,20,64)
check('rope_cos', rope_cos[:20].unsqueeze(0), cos_ref.float())
check('rope_sin', rope_sin[:20].unsqueeze(0), sin_ref.float())

# ═══ Test 1: full attention prefill ═══
S=14
x=torch.randn(1,S,H,device=DEV,dtype=torch.bfloat16)
am=Qwen3_5MoeAttention(cfg,layer_idx=3).to(DEV).to(torch.bfloat16).eval()
pidsS=torch.arange(S,device=DEV).view(1,1,-1).expand(3,1,-1)
cos,sin=rope_mod(x,pidsS)
causal4=torch.triu(torch.full((S,S),float('-inf'),device=DEV),diagonal=1).view(1,1,S,S)
with torch.no_grad():
    ref_out,_=am(x,(cos,sin),attention_mask=causal4,past_key_values=None)
gpu={
 'self_attn.q_proj.weight':am.q_proj.weight,'self_attn.k_proj.weight':am.k_proj.weight,
 'self_attn.v_proj.weight':am.v_proj.weight,'self_attn.o_proj.weight':am.o_proj.weight,
 'self_attn.q_norm.weight':am.q_norm.weight,'self_attn.k_norm.weight':am.k_norm.weight,
}
with torch.no_grad():
    mine_out=attn(x,gpu,3,'full_attention',True,{})
check('full_attention prefill', mine_out, ref_out)

# ═══ Test 2: linear attention prefill + decode ═══
x2=torch.randn(1,S,H,device=DEV,dtype=torch.bfloat16)
ln=Qwen3_5MoeGatedDeltaNet(cfg,layer_idx=0).to(DEV).to(torch.bfloat16).eval()
with torch.no_grad():
    ref_prefill=ln(x2,cache_params=None)
gpu2={
 'linear_attn.in_proj_qkv.weight':ln.in_proj_qkv.weight,'linear_attn.in_proj_z.weight':ln.in_proj_z.weight,
 'linear_attn.in_proj_a.weight':ln.in_proj_a.weight,'linear_attn.in_proj_b.weight':ln.in_proj_b.weight,
 'linear_attn.out_proj.weight':ln.out_proj.weight,'linear_attn.conv1d.weight':ln.conv1d.weight,
 'linear_attn.dt_bias':ln.dt_bias,'linear_attn.A_log':ln.A_log,'linear_attn.norm.weight':ln.norm.weight,
}
cache={}
with torch.no_grad():
    mine_prefill=attn(x2,gpu2,0,'linear_attention',True,cache)
check('linear_attn prefill', mine_prefill, ref_prefill)

# --- decode 1 token: 官方 = conv_update + recurrent; 我 = conv拼接 + chunk(S=1, init=state) ---
x3=torch.randn(1,1,H,device=DEV,dtype=torch.bfloat16)
with torch.no_grad():
    # 官方参考: 手工复现官方 decode 路径
    hs=x3
    mixed=ln.in_proj_qkv(hs).transpose(1,2)
    # 官方 prefill 时的 conv_state = 原始输入最后(补到kernel长度)
    # 先重新算 prefill 的 conv 输入
    mixed_pre=ln.in_proj_qkv(x2).transpose(1,2)
    conv_state=F.pad(mixed_pre,(ln.conv_kernel_size-mixed_pre.shape[-1],0))[:,:,:ln.conv_kernel_size]
    conv_state=conv_state.contiguous()
    mixed_dec=ln.causal_conv1d_update(mixed,conv_state,ln.conv1d.weight.squeeze(1),ln.conv1d.bias,ln.activation)
    z=ln.in_proj_z(hs).reshape(1,1,-1,ln.head_v_dim)
    b=ln.in_proj_b(hs);a=ln.in_proj_a(hs)
    mixed_dec=mixed_dec.transpose(1,2)
    KD=ln.key_dim
    query,key,value=torch.split(mixed_dec,[KD,KD,ln.value_dim],dim=-1)
    query=query.reshape(1,1,-1,ln.head_k_dim);key=key.reshape(1,1,-1,ln.head_k_dim);value=value.reshape(1,1,-1,ln.head_v_dim)
    beta=b.sigmoid()
    g=-ln.A_log.float().exp()*F.softplus(a.float()+ln.dt_bias)
    query=query.repeat_interleave(NV//NK,dim=2);key=key.repeat_interleave(NV//NK,dim=2)
    # prefill 的 recurrent state (用官方chunk, output_final_state=True)
    q_pre,k_pre,v_pre=None,None,None
    mp=ln.in_proj_qkv(x2).transpose(1,2)
    mp=F.silu(ln.conv1d(mp)[:, :, :mp.shape[-1]]).transpose(1,2)
    qp,kp,vp=torch.split(mp,[KD,KD,ln.value_dim],dim=-1)
    qp=qp.reshape(1,S,-1,ln.head_k_dim);kp=kp.reshape(1,S,-1,ln.head_k_dim);vp=vp.reshape(1,S,-1,ln.head_v_dim)
    bp=ln.in_proj_b(x2).sigmoid();ap=ln.in_proj_a(x2)
    gp=-ln.A_log.float().exp()*F.softplus(ap.float()+ln.dt_bias)
    qp=qp.repeat_interleave(NV//NK,dim=2);kp=kp.repeat_interleave(NV//NK,dim=2)
    _,rec_state=torch_chunk_gated_delta_rule(qp,kp,vp,gp,bp,64,None,True,True)
    dec_ref,_=torch_recurrent_gated_delta_rule(query,key,value,g,beta,rec_state,True,True)
    core=dec_ref.reshape(-1,ln.head_v_dim)
    zz=z.reshape(-1,ln.head_v_dim)
    core=ln.norm(core,zz).reshape(1,1,-1)
    ref_decode=ln.out_proj(core)
with torch.no_grad():
    mine_decode=attn(x3,gpu2,0,'linear_attention',False,cache)
check('linear_attn decode', mine_decode, ref_decode)

# ═══ Test 3: MoE block ═══
import copy
cfg_moe=copy.deepcopy(cfg);cfg_moe.num_experts=8
moe=Qwen3_5MoeSparseMoeBlock(cfg_moe).to(DEV).to(torch.bfloat16).eval()
xm=torch.randn(1,S,H,device=DEV,dtype=torch.bfloat16)
with torch.no_grad():
    ref_moe=moe(xm)
# 手写 MoE (与 test_122b_fixed.py 相同逻辑)
flat=xm.reshape(-1,H)
rw=moe.gate.weight
probs=torch.softmax(flat.float()@rw.float().T,-1);tw,ti=torch.topk(probs,8,-1);tw=tw/tw.sum(-1,keepdim=True)
uids=ti.unique().tolist()
out=torch.zeros_like(flat,dtype=torch.float32)
ge=moe.experts.gate_up_proj;de=moe.experts.down_proj
for eid in uids:
    g=ge[eid].float().to(DEV);d=de[eid].float().to(DEV);mask=(ti==eid).any(-1)
    if mask.any():
        t2=flat[mask].float();eid_mask=(ti[mask]==eid);gw=(tw[mask]*eid_mask).sum(-1)
        out[mask]+=gw.unsqueeze(-1)*((F.silu(t2@g[:1024].T)*(t2@g[1024:].T))@d.T)
sgw=moe.shared_expert_gate.weight
sg=moe.shared_expert.gate_proj.weight;su=moe.shared_expert.up_proj.weight;sd=moe.shared_expert.down_proj.weight
gv=torch.sigmoid(flat.float()@sgw.float().T)
so=(F.silu(flat.float()@sg.float().T)*(flat.float()@su.float().T))@sd.float().T
out=out+gv*so
mine_moe=out.reshape(1,S,H).to(torch.bfloat16)
check('moe block', mine_moe, ref_moe)

# ═══ Test 4: norms ═══
n1=Qwen3_5MoeRMSNorm(H,eps=EP).to(DEV).eval()
xn=torch.randn(1,S,H,device=DEV,dtype=torch.bfloat16)
with torch.no_grad():ref_n=n1(xn)
mine_n=(xn.float()*torch.rsqrt(xn.float().pow(2).mean(-1,keepdim=True)+EP)*(1.0+n1.weight.float())).to(torch.bfloat16)
check('rmsnorm(1+w)', mine_n, ref_n)
n2=Qwen3_5MoeRMSNormGated(HV,eps=EP).to(DEV).eval()
xg=torch.randn(S*NV,HV,device=DEV,dtype=torch.bfloat16);gate=torch.randn(S*NV,HV,device=DEV,dtype=torch.bfloat16)
with torch.no_grad():ref_g=n2(xg,gate)
mine_g=(xg.float()*torch.rsqrt(xg.float().pow(2).mean(-1,keepdim=True)+EP)*n2.weight.float()*F.silu(gate.float())).to(torch.bfloat16)
check('rmsnorm_gated', mine_g, ref_g)

print('\n=== SUMMARY ===')
allok=all(r[1] for r in results)
for n,ok,mx,re in results:print(f'{"PASS" if ok else "FAIL"} {n} (rel={re:.5f})')
print('ALL PASS' if allok else 'SOME FAILED')
