"""
122B MoE vFIXED: 对照官方 transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py 修正
修复清单:
 1. 补上 final RMSNorm (model.language_model.norm.weight) + 加载它
 2. full_attn q/gate 按头拆分: view(B,S,32,512).chunk(2,-1), 不是平铺对半
 3. q_norm 只作用于 query, 不作用于 gate
 4. MoE 遍历全部命中专家 (去掉 uids[:8] 截断)
 5. linear_attn.norm 直接乘 w (RMSNormGated, ones初始化), 不是 (1+w)
 6. decode RoPE 位置 = prompt_len + step (cur_pos), 不是从0数
 7. conv1d 缓存原始输入(卷积前); decode 用 padding=0 只取当前位置输出
 8. KV cache 直接整体写入, 不重复拼接, 不截断
 9. 验证阶段改用 greedy 解码, 结果可复现
"""
import torch, torch.nn.functional as F, time, json, os, sys
DEV='cuda';sys.path.insert(0,r'D:\ai')
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule, l2norm  # 同上
from expert_reader import SafeExpertReader

MODEL=r'D:\ai\models\Qwen3.5-122B-A10B'

tk=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
try:
    cf=AutoConfig.from_pretrained(MODEL,trust_remote_code=True).text_config
except Exception:
    import types as _ty
    cf=_ty.SimpleNamespace(**json.load(open(MODEL+'/config.json',encoding='utf-8'))['text_config'])  # 4.57.6不认识qwen3_5_moe, 直读json
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types;EP=cf.rms_norm_eps
NV,NK,HK,HV=64,16,128,128
print(f'122B: {cf.num_hidden_layers} layers, H={H}, V={V}')

# ═══ load ALL non-expert tensors via safe_open (逐个分片, 只取需要的键, 读完即关) ═══
print('Loading non-expert...',end='',flush=True);t0=time.perf_counter()
idx=json.load(open(MODEL+'/model.safetensors.index.json'))
layer_ram=[{} for _ in range(48)]
emb=None;lm_head=None;final_norm=None;loaded=0
# 按分片分组需要的非专家键
shard_keys={}
for name,shard in idx['weight_map'].items():
    if '.mlp.experts.' in name or name.startswith('mtp.') or name.startswith('model.visual'):continue
    shard_keys.setdefault(shard,[]).append(name)
for shard,names in shard_keys.items():
    print(f'\n  [load] {shard} ({len(names)} keys)',end='',flush=True)
    f=safe_open(MODEL+'/'+shard,framework='pt')
    for k in names:
        v=f.get_tensor(k).clone()   # clone = 拷贝进内存, 不依赖 mmap
        loaded+=1;name=k.replace('model.language_model.','')
        if 'embed_tokens' in name:emb=v
        elif 'lm_head' in name:lm_head=v
        elif name=='norm.weight':final_norm=v   # FIX 1: final norm
        elif 'layers.' in name:
            p=name.split('.');lid=int(p[p.index('layers')+1]);key='.'.join(p[p.index('layers')+2:])
            layer_ram[lid][key]=v
    del f   # 关闭当前分片的映射再开下一个
    print(' done',end='',flush=True)
assert final_norm is not None and emb is not None and lm_head is not None, 'missing weights!'
# 所有分片关闭后再做 GPU 转移 (映射循环内做 H2D 会触发本机环境段错误)
emb=emb.float().to(DEV);lm_head=lm_head.float().to(DEV);final_norm=final_norm.float().to(DEV)
print('\n  [load] emb/lm_head/final_norm -> GPU ok',flush=True)
gb=sum(sum(vv.numel()*vv.element_size() for vv in d.values()) for d in layer_ram)/1024**3
print(f' {time.perf_counter()-t0:.0f}s ({gb:.1f}GB RAM, {loaded} non-exp tensors)',flush=True)

rd=SafeExpertReader(MODEL)
print('[init] reader ready',flush=True)

# ═══ KV Cache ═══
class KVC:
    def __init__(self):self.kv={};self.cv={};self.rc={}
    def gkv(self,l):return self.kv.get(l)
    def gcv(self,l):return self.cv.get(l)
    def grc(self,l):return self.rc.get(l)
    def skv(self,l,k,v):self.kv[l]=(k,v)        # FIX 8: 整体写入
    def ucv(self,l,s):self.cv[l]=s
    def urc(self,l,s):self.rc[l]=s

cur_pos=0  # FIX 6: 下一个 decode token 的 RoPE 位置

# ═══ Attention ═══
def attn(h,gpu,lid,bt,pre,cache):
    B,S,_=h.shape
    def w(k):v=gpu.get(k);return v.float().to(DEV) if v is not None else None
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        qn=gpu.get('self_attn.q_norm.weight');kn=gpu.get('self_attn.k_norm.weight')
        if q is None:return h*0.99
        # FIX 2: 官方布局 = 每头 [q(256)|gate(256)], view(B,S,32,512) 后按最后一维拆
        Q=(h.float()@q.T).view(B,S,32,512)
        qa=Q[:,:,:,:256];qg=Q[:,:,:,256:]
        if qn is not None:  # FIX 3: 官方只对 query 做 q_norm, gate 不过 norm
            qa=qa*torch.rsqrt(qa.pow(2).mean(-1,keepdim=True)+EP)*(1.0+qn.float().to(DEV))
        qa=qa.transpose(1,2);qg=qg.transpose(1,2)
        K=(h.float()@k.T).view(B,S,2,256)
        Vm=(h.float()@v.T).view(B,S,2,256)
        if kn is not None:K=K*torch.rsqrt(K.pow(2).mean(-1,keepdim=True)+EP)*(1.0+kn.float().to(DEV))
        K=K.transpose(1,2).repeat_interleave(16,1);Vm=Vm.transpose(1,2).repeat_interleave(16,1)
        # RoPE: rotate first rope_dim channels (rotate_half 语义, 与官方一致)
        Hq = qa.shape[1]
        pids = torch.arange(S, device=DEV) if pre else torch.tensor([cur_pos], device=DEV)  # FIX 6
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
            kv=cache.gkv(lid)
            if kv is not None:K=torch.cat([kv[0],K],2);Vm=torch.cat([kv[1],Vm],2)
        cache.skv(lid,K,Vm)
        att=F.scaled_dot_product_attention(qa,K,Vm,is_causal=pre)
        gated=torch.sigmoid(qg)*att
        if lid<5:print(f'L{lid} gate_sig={torch.sigmoid(qg).mean():.3f} att_n={att.norm():.1f} gated_n={gated.norm():.1f}',flush=True)
        return (gated.transpose(1,2).contiguous().view(B,S,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight');a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight');o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None:return h*0.99
        KD=NK*HK;VD=NV*HV;m=h.float()@qkv.T;m=m.transpose(1,2)
        if cv is not None:
            cw=cv.float()
            if cw.dim()==2:cw=cw.unsqueeze(1)
            # FIX 7: 缓存卷积前原始输入的最后3个; decode 用 padding=0 产出当前1个位置
            if not pre:
                cs=cache.gcv(lid)
                if cs is not None:
                    mcat=torch.cat([cs,m],-1)
                    cache.ucv(lid,mcat[:,:,-3:].clone())
                    m=F.silu(F.conv1d(mcat,cw,padding=0,groups=12288))
                else:
                    cache.ucv(lid,F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone())
                    m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
            else:
                cache.ucv(lid,F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone())
                m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);V=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h.float()@b.T).sigmoid() if b is not None else torch.ones(B,S,NV,device=DEV)
        av=(h.float()@a.T) if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+(dt.float() if dt is not None else 0))
        R=NV//NK;Q=Q.repeat_interleave(R,2);K=K.repeat_interleave(R,2)   # FIX 10: 头映射须与官方一致(h//R), expand+reshape 是 h%NK
        init=None if pre else cache.grc(lid)
        ca,ls=torch_chunk_gated_delta_rule(Q,K,V,g,beta,64,init,True,True);cache.urc(lid,ls)
        at=ca.float().contiguous().view(B,S,VD)   # FIX 11: kernel 返回已是 (B,S,NV,HV), 不能再 transpose
        if nr is not None:
            ni=at.reshape(B,S,NV,HV)
            # FIX 5: RMSNormGated 直接乘 weight (ones初始化), 不是 (1+w)
            at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.view(1,1,1,-1)).reshape(B,S,VD)
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
MAXP=4096
rope_cos = torch.zeros(MAXP, rope_dim, device=DEV)
rope_sin = torch.zeros(MAXP, rope_dim, device=DEV)
for p in range(MAXP):
    theta = p * freq
    rope_cos[p] = torch.cat([theta.cos(), theta.cos()])
    rope_sin[p] = torch.cat([theta.sin(), theta.sin()])
cur_pos = ids.shape[1]   # FIX 6: 第一个生成 token 的位置 = prompt 长度
print(f'[init] prompt {cur_pos} tokens, rope ready, start gen',flush=True)
t0=time.perf_counter();gen=[]
for step in range(32):
    pre=(step==0)
    print(f'[step {step}] begin',flush=True)
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
        for eid in uids:   # FIX 4: 全部命中专家都算
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
    # FIX 1: final RMSNorm (1+w), 官方 Qwen3_5MoeTextModel 最后一步
    hf=h.float()
    hnorm=hf*torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+EP)*(1.0+final_norm)
    logits=hnorm[:,-1:,:]@lm_head.T
    nt=int(logits[0,-1].argmax().item())   # FIX 9: greedy 验证
    gen.append(nt)
    print(f'[{step}] {tk.decode([nt])}',flush=True)
    if nt==tk.eos_token_id:break
    h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(torch.bfloat16)
    if not pre:cur_pos+=1

print(f'\nOutput: {tk.decode(gen,skip_special_tokens=True)[:300]}')
print(f'{time.perf_counter()-t0:.0f}s VRAM:{torch.cuda.max_memory_allocated()/1024**3:.1f}GB')
rd.close()
