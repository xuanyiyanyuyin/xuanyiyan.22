"""
122B MoE vFAST: 在 test_122b_fixed.py(已验证正确)基础上的性能优化版
- 全程 bf16 计算(与官方数值行为一致), 只在 norm/router 处用 f32
- emb/lm_head bf16 常驻 GPU (3GB, 原 f32 要 6GB)
- 专家权重: bytes->frombuffer->bf16->直接上GPU, 不做 CPU clone, 不转 f32
- 层权重 bf16 流式上传, 用独立 CUDA stream 提前一层预取
- 每步打印 tok/s
"""
import torch, torch.nn.functional as F, time, json, os, sys, struct, ctypes
DEV='cuda';sys.path.insert(0,r'D:\ai')
# 低优先级运行, 避免抢占用户前台资源
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
except Exception:pass
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule  # 4.57.6无qwen3_5_moe, 内核逐字相同
import threading, warnings
warnings.filterwarnings('ignore', message='.*buffer is not writable.*')

MODEL=r'D:\ai\models\Qwen3.5-122B-A10B'

# ═══ 纯文件读取加载(不用mmap, 绕开本机 mmap 段错误问题) ═══
DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool}
def load_keys_pread(path,wanted):
    out={}
    with open(path,'rb') as f:
        hs=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(hs))
        base=8+hs
        for k in wanted:
            meta=header[k]
            o0,o1=meta['data_offsets']
            f.seek(base+o0)
            buf=bytearray(f.read(o1-o0))
            out[k]=torch.frombuffer(buf,dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
    return out

tk=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
try:
    cf=AutoConfig.from_pretrained(MODEL,trust_remote_code=True).text_config
except Exception:
    import types as _ty
    cf=_ty.SimpleNamespace(**json.load(open(MODEL+'/config.json',encoding='utf-8'))['text_config'])  # 4.57.6不认识qwen3_5_moe, 直读json
H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types;EP=cf.rms_norm_eps
NV,NK,HK,HV=64,16,128,128
print(f'122B: {cf.num_hidden_layers} layers, H={H}, V={V}',flush=True)

# ═══ 加载非专家权重(bf16, 全部读完关闭文件后再上GPU) ═══
print('Loading non-expert...',end='',flush=True);t0=time.perf_counter()
idx=json.load(open(MODEL+'/model.safetensors.index.json'))
layer_ram=[{} for _ in range(48)]
emb=None;lm_head=None;final_norm=None;loaded=0
shard_keys={}
for name,shard in idx['weight_map'].items():
    if '.mlp.experts.' in name or name.startswith('mtp.') or name.startswith('model.visual'):continue
    shard_keys.setdefault(shard,[]).append(name)
for shard,names in shard_keys.items():
    print(f'\n  [load] {shard} ({len(names)} keys)',end='',flush=True)
    tensors=load_keys_pread(MODEL+'/'+shard,names)
    for k,v in tensors.items():
        loaded+=1;name=k.replace('model.language_model.','')
        if 'embed_tokens' in name:emb=v
        elif 'lm_head' in name:lm_head=v
        elif name=='norm.weight':final_norm=v
        elif 'layers.' in name:
            p=name.split('.');lid=int(p[p.index('layers')+1]);key='.'.join(p[p.index('layers')+2:])
            layer_ram[lid][key]=v
    del tensors
    print(' done',end='',flush=True)
print('\n  [load] files closed, moving to GPU...',end='',flush=True)
emb=emb.to(DEV);lm_head=lm_head.to(DEV);final_norm=final_norm.float().to(DEV)
torch.cuda.synchronize()
print(f' {time.perf_counter()-t0:.0f}s ({loaded} tensors, emb/lm_head bf16 on GPU)',flush=True)

# ═══ 专家读取: pread版(无mmap, 线程本地文件句柄, 支持多盘路径重映射) ═══
class PreadExpertReader:
    def __init__(self,model_dir):
        self.model_dir=model_dir
        idx=json.load(open(model_dir+'/model.safetensors.index.json'))
        self._locs={};self.paths={}
        for name,shard in idx['weight_map'].items():
            if 'experts' in name and 'shared' not in name and not name.startswith('mtp'):
                parts=name.split('.')
                if parts[-1] in ('gate_up_proj','down_proj'):
                    lid=int(parts[parts.index('layers')+1])
                    self._locs[(lid,parts[-1])]=shard
                    self.paths[shard]=model_dir+'/'+shard
        self._hdr={};self._tl=threading.local()
        n_exp=sum(1 for k in self._locs)//2
        print(f'[PreadExpertReader] {n_exp} layers, {len(self.paths)} shards',flush=True)
    def remap(self,shard,newpath):self.paths[shard]=newpath;self._hdr.pop(shard,None)
    def _fh(self,shard):
        hs=getattr(self._tl,'h',None)
        if hs is None:hs={};self._tl.h=hs
        if shard not in hs:hs[shard]=open(self.paths[shard],'rb')
        return hs[shard]
    def _header(self,shard):
        hd=self._hdr.get(shard)
        if hd is None:
            with open(self.paths[shard],'rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                hd=(n+8,json.loads(f.read(n)))
            self._hdr[shard]=hd
        return hd
    def read_expert(self,layer,eid):
        out=[]
        for proj in ('gate_up_proj','down_proj'):
            shard=self._locs[(layer,proj)]
            base,hdr=self._header(shard)
            meta=hdr[f'model.language_model.layers.{layer}.mlp.experts.{proj}']
            o0,o1=meta['data_offsets'];n=meta['shape'][0];per=(o1-o0)//n
            f=self._fh(shard)
            f.seek(base+o0+eid*per)
            buf=f.read(per)   # bytes: 该线程独享, frombuffer直读
            out.append(torch.frombuffer(buf,dtype=torch.bfloat16).reshape(meta['shape'][1:]))
        return out[0],out[1]
    def close(self):pass

rd=PreadExpertReader(MODEL)
# 双盘搬运: 部分专家分片放C盘, 两NVMe并行读
import os.path
for _s in ('model.safetensors-00002-of-00039.safetensors','model.safetensors-00005-of-00039.safetensors',
           'model.safetensors-00008-of-00039.safetensors','model.safetensors-00011-of-00039.safetensors',
           'model.safetensors-00014-of-00039.safetensors','model.safetensors-00017-of-00039.safetensors',
           'model.safetensors-00026-of-00039.safetensors','model.safetensors-00029-of-00039.safetensors',
           'model.safetensors-00032-of-00039.safetensors'):
    _p='C:/ai_experts/'+_s
    if os.path.exists(_p):rd.remap(_s,_p)
print(f'[dual-disk] {sum(1 for p in rd.paths.values() if p.startswith("C:"))} shards on C:, rest on D:',flush=True)

# ═══ KV/状态缓存 (bf16/f32) ═══
class KVC:
    def __init__(self):self.kv={};self.cv={};self.rc={}
    def gkv(self,l):return self.kv.get(l)
    def gcv(self,l):return self.cv.get(l)
    def grc(self,l):return self.rc.get(l)
    def skv(self,l,k,v):self.kv[l]=(k,v)
    def ucv(self,l,s):self.cv[l]=s
    def urc(self,l,s):self.rc[l]=s

cur_pos=0
copy_stream=torch.cuda.Stream()

# ═══ 层权重预取: 提前一层把 bf16 权重搬上GPU ═══
_pref={}
_pref_lid=-1
def prefetch_layer(lid):
    global _pref,_pref_lid
    with torch.cuda.stream(copy_stream):
        _pref={k:v.to(DEV,non_blocking=True) for k,v in layer_ram[lid].items()}
    _pref_lid=lid
def get_layer(lid):
    torch.cuda.current_stream().wait_stream(copy_stream)
    assert _pref_lid==lid, f'prefetch mismatch: have {_pref_lid} want {lid}'
    return _pref

# ═══ Attention (bf16) ═══
def attn(h,gpu,lid,bt,pre,cache):
    B,S,_=h.shape
    def w(k):return gpu.get(k)
    if bt=='full_attention':
        q=w('self_attn.q_proj.weight');k=w('self_attn.k_proj.weight');v=w('self_attn.v_proj.weight');o=w('self_attn.o_proj.weight')
        qn=gpu.get('self_attn.q_norm.weight');kn=gpu.get('self_attn.k_norm.weight')
        if q is None:return h*0.99
        Q=(h@q.T).view(B,S,32,512)
        qa=Q[:,:,:,:256].float();qg=Q[:,:,:,256:]
        if qn is not None:
            qa=qa*torch.rsqrt(qa.pow(2).mean(-1,keepdim=True)+EP)*(1.0+qn.float())
        qa=qa.to(torch.bfloat16).transpose(1,2);qg=qg.transpose(1,2)
        K=(h@k.T).view(B,S,2,256).float();Vm=(h@v.T).view(B,S,2,256)
        if kn is not None:K=K*torch.rsqrt(K.pow(2).mean(-1,keepdim=True)+EP)*(1.0+kn.float())
        K=K.to(torch.bfloat16).transpose(1,2).repeat_interleave(16,1);Vm=Vm.transpose(1,2).repeat_interleave(16,1)
        Hq=qa.shape[1]
        pids = torch.arange(S, device=DEV) if pre else torch.tensor([cur_pos], device=DEV)
        c=rope_cos[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        s=rope_sin[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        x_rot=qa[:,:,:,:rope_dim]
        x1,x2=x_rot[...,:rope_dim//2],x_rot[...,rope_dim//2:]
        c1,c2=c[...,:rope_dim//2],c[...,rope_dim//2:]
        s1,s2=s[...,:rope_dim//2],s[...,rope_dim//2:]
        rotated=torch.cat([x1*c1-x2*s1,x2*c2+x1*s2],dim=-1)
        qa=torch.cat([rotated,qa[:,:,:,rope_dim:]],dim=-1)
        Kr=K[:,:,:,:rope_dim]
        K1,K2=Kr[...,:rope_dim//2],Kr[...,rope_dim//2:]
        K_rot=torch.cat([K1*c1-K2*s1,K2*c2+K1*s2],dim=-1)
        K=torch.cat([K_rot,K[:,:,:,rope_dim:]],dim=-1)
        if not pre:
            kv=cache.gkv(lid)
            if kv is not None:K=torch.cat([kv[0],K],2);Vm=torch.cat([kv[1],Vm],2)
        cache.skv(lid,K,Vm)
        att=F.scaled_dot_product_attention(qa,K,Vm,is_causal=pre)
        gated=torch.sigmoid(qg)*att
        return gated.transpose(1,2).contiguous().view(B,S,8192)@o.T
    else:
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight');a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight');o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None:return h*0.99
        KD=NK*HK;VD=NV*HV;m=(h@qkv.T).transpose(1,2)
        if cv is not None:
            cw=cv
            if cw.dim()==2:cw=cw.unsqueeze(1)
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
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);Vm=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h@b.T).float().sigmoid() if b is not None else torch.ones(B,S,NV,device=DEV)
        av=(h@a.T).float() if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+dt.float())
        R=NV//NK;Q=Q.repeat_interleave(R,2);K=K.repeat_interleave(R,2)
        init=None if pre else cache.grc(lid)
        if S==1:
            ca,ls=torch_recurrent_gated_delta_rule(Q,K,Vm,g,beta,init,True,True)   # 官方decode路径, O(1)
        else:
            ca,ls=torch_chunk_gated_delta_rule(Q,K,Vm,g,beta,64,init,True,True)
        cache.urc(lid,ls)
        at=ca.float().contiguous().view(B,S,VD)
        if nr is not None:
            ni=at.reshape(B,S,NV,HV)
            at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.float().view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h@z.T).float() if z is not None else at
        at=(at*F.silu(zz)).to(torch.bfloat16)
        return (at@o.T) if o is not None else at[:,:,:H]

# ═══ INFERENCE ═══
PROMPT='请用中文介绍你自己'
print(f'\n"{PROMPT}"',flush=True)
ids=tk.apply_chat_template([{'role':'user','content':PROMPT}],add_generation_prompt=True,return_tensors='pt',tokenize=True).input_ids.to(DEV)
h=F.embedding(ids,emb);cache=KVC()
rope_dim=int(256*0.25)
freq=1.0/(10000000**(torch.arange(0,rope_dim,2,device=DEV).float()/rope_dim))
MAXP=4096
rope_cos=torch.zeros(MAXP,rope_dim,device=DEV,dtype=torch.bfloat16)
rope_sin=torch.zeros(MAXP,rope_dim,device=DEV,dtype=torch.bfloat16)
for p in range(MAXP):
    theta=p*freq
    rope_cos[p]=torch.cat([theta.cos(),theta.cos()]).to(torch.bfloat16)
    rope_sin[p]=torch.cat([theta.sin(),theta.sin()]).to(torch.bfloat16)
cur_pos=ids.shape[1]
print(f'[init] prompt {cur_pos} tokens',flush=True)

# 专家预读线程池 (16线程并发, 用户CPU 20线程)
from concurrent.futures import ThreadPoolExecutor
pool=ThreadPoolExecutor(max_workers=16)
spec_pool=ThreadPoolExecutor(max_workers=4)   # 跨层投机预取专用, 不挤占按需读取

# 专家 RAM LRU 缓存 (内存搬运: 命中直接上GPU, 不读盘)
from collections import OrderedDict
exp_cache=OrderedDict();exp_lock=threading.Lock()
EXP_CACHE_MAX=650   # ~12GB RAM, 给系统留足余量
cache_hits=[0];cache_miss=[0]
def get_expert(lid,eid):
    key=(lid,eid)
    with exp_lock:
        hit=exp_cache.get(key)
        if hit is not None:
            exp_cache.move_to_end(key)
            cache_hits[0]+=1
            return hit
    g,d=rd.read_expert(lid,eid)
    with exp_lock:
        cache_miss[0]+=1
        exp_cache[key]=(g,d)
        while len(exp_cache)>EXP_CACHE_MAX:exp_cache.popitem(last=False)
    return g,d

xfer_stream=torch.cuda.Stream()   # 专家传输专用流, 与计算重叠(王牌1: batch并行传输)

def moe_forward(flat,gpu,lid,prefill_tokens):
    rw=gpu.get('mlp.gate.weight')
    probs=torch.softmax(flat.float()@rw.float().T,-1)
    tw,ti=torch.topk(probs,8,-1);tw=tw/tw.sum(-1,keepdim=True)
    uids=ti.unique().tolist()
    out=torch.zeros(flat.shape[0],H,device=DEV,dtype=torch.float32)
    # 并行取专家 (RAM缓存命中则零读盘)
    futs={eid:pool.submit(get_expert,lid,eid) for eid in uids}
    gs=[];ds=[];oks=[]
    for eid in uids:
        try:gn,dn=futs[eid].result()
        except:continue
        gs.append(gn);ds.append(dn);oks.append(eid)
    E=len(oks)
    if E>0:
        if flat.shape[0]==1:
            # decode(8专家): 堆叠大张量一次H2D + 批量bmm
            G=torch.stack(gs);D=torch.stack(ds)
            with torch.cuda.stream(xfer_stream):
                Gd=G.to(DEV,non_blocking=True);Dd=D.to(DEV,non_blocking=True)
        else:
            # prefill(专家多): 逐专家传输(默认流序保证), 避免巨型CPU堆叠拷贝
            Gd=[g.to(DEV,non_blocking=True) for g in gs]
            Dd=[d.to(DEV,non_blocking=True) for d in ds]
    # 共享专家: 在专家传输期间于默认流上计算(天然重叠)
    sgw=gpu.get('mlp.shared_expert_gate.weight')
    if sgw is not None:
        sg=gpu.get('mlp.shared_expert.gate_proj.weight');su=gpu.get('mlp.shared_expert.up_proj.weight');sd=gpu.get('mlp.shared_expert.down_proj.weight')
        if sg is not None:
            gv=torch.sigmoid(flat.float()@sgw.float().T)
            out=out+gv*((F.silu(flat@sg.T)*(flat@su.T))@sd.T).float()
    if E>0:
        if flat.shape[0]==1:
            torch.cuda.current_stream().wait_stream(xfer_stream)
            gateup=torch.einsum('th,edh->ted',flat,Gd)          # (1,E,2048)
            inter=F.silu(gateup[:,:,:1024])*gateup[:,:,1024:]   # (1,E,1024)
            eout=torch.einsum('tei,eoi->teo',inter,Dd)          # (1,E,3072)
            wvec=torch.stack([(tw[0][ti[0]==eid]).sum() for eid in oks]).to(DEV)
            out=out+(eout[0].float()*wvec.unsqueeze(-1)).sum(0,keepdim=True)
        else:
            for j,eid in enumerate(oks):
                mask=(ti==eid).any(-1)
                if mask.any():
                    t2=flat[mask];g=Gd[j];d=Dd[j]
                    gw=(tw[mask].float()*(ti[mask]==eid)).sum(-1)
                    eout=((F.silu(t2@g[:1024].T)*(t2@g[1024:].T))@d.T).float()
                    out[mask]+=gw.unsqueeze(-1)*eout
    return out,uids

t0=time.perf_counter();gen=[];tok_times=[]
prefetch_layer(0)
for step in range(32):
    pre=(step==0)
    ts=time.perf_counter();t_attn=0.0;t_moe=0.0
    for lid in range(48):
        bt=LV[lid]
        gpu=get_layer(lid)
        if lid+1<48:prefetch_layer(lid+1)
        else:prefetch_layer(0)   # 环绕: 算47层时为下一个token预取第0层
        inn=gpu.get('input_layernorm.weight')
        hn=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None:hn=hn*(1.0+inn.float())
        hn=hn.to(torch.bfloat16)
        ta=time.perf_counter()
        a_out=attn(hn,gpu,lid,bt,pre,cache)
        tb=time.perf_counter();t_attn+=tb-ta
        h=h+a_out
        pn=gpu.get('post_attention_layernorm.weight')
        hn2=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None:hn2=hn2*(1.0+pn.float())
        flat=hn2.to(torch.bfloat16).reshape(-1,H)
        out,uids=moe_forward(flat,gpu,lid,pre)
        t_moe+=time.perf_counter()-tb
        h=(h+out.to(torch.bfloat16).reshape_as(h))
        del gpu
        if pre and (lid%8==0):print(f'  prefill L{lid:02d} {time.perf_counter()-ts:.0f}s',flush=True)
    hf=h.float()
    hnorm=hf*torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+EP)*(1.0+final_norm)
    logits=(hnorm[:,-1:,:].to(torch.bfloat16)@lm_head.T).float()   # bf16矩阵乘, 省3GB f32显存
    nt=int(logits[0,-1].argmax().item())
    gen.append(nt)
    dt_=time.perf_counter()-ts;tok_times.append(dt_)
    hitrate=cache_hits[0]/max(1,cache_hits[0]+cache_miss[0])
    print(f'[{step}] {tk.decode([nt])!r}  ({dt_:.1f}s attn={t_attn:.1f} moe={t_moe:.1f} cache={hitrate:.0%})',flush=True)
    if nt==tk.eos_token_id:break
    h=F.embedding(torch.tensor([[nt]],device=DEV),emb)
    if not pre:cur_pos+=1

print(f'\nOutput: {tk.decode(gen,skip_special_tokens=True)[:400]}',flush=True)
if len(tok_times)>1:
    dec=tok_times[1:]
    print(f'prefill {tok_times[0]:.1f}s | decode avg {sum(dec)/len(dec):.2f}s/tok ({len(dec)/sum(dec):.2f} tok/s)',flush=True)
print(f'total {time.perf_counter()-t0:.0f}s VRAM:{torch.cuda.max_memory_allocated()/1024**3:.1f}GB',flush=True)
rd.close()
