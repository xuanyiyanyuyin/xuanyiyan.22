"""
Qwen3.5-122B-A10B 本地聊天服务
- 零依赖(纯标准库), SSE 流式输出
- 引擎 = test_122b_mem.py 的已验证实现(读写头上下文 + 窗口KV)
- 启动: D:\ai\python\python.exe qwen_server.py
- 打开: http://127.0.0.1:8122
"""
import torch, torch.nn.functional as F, time, json, os, sys, struct, threading
DEV='cuda';sys.path.insert(0,r'D:\ai')
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule  # 同上
from memory_head import MemoryHead
import http.server, socketserver

MODEL=r'D:\ai\models\Qwen3.5-122B-A10B'
PORT=8122
WINDOW=128       # 原始KV窗口
MEM_TOPK=16      # 每层召回记忆数 (覆盖重要主体)
FLUSH_SEG=1      # 每1token存一条记忆 (无压缩, 存得慢但多)
MAX_NEW_DEFAULT=100000  # 无限制, 模型思考到EOS自然停(122B本来慢, 不截断)

tk=None;cf=None;H=None;V=None;LV=None;EP=None
NV,NK,HK,HV=64,16,128,128
layer_ram=None;emb=None;lm_head=None;final_norm=None
rd=None
rope_cos=None;rope_sin=None;rope_dim=64
mem_heads={};pend_h={};pend_pos={}
ENGINE_READY=False
LOCK=threading.Lock()
STATS={'tokens':0,'turns':0,'started':None}

# ═══ 专家读取(pread, 线程本地fd) ═══
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
            buf=f.read(per)
            # ⚠️ 必须拷贝: frombuffer 只引用 bytes 内存, buf 释放后是悬空指针!
            t=torch.frombuffer(buf,dtype=torch.bfloat16).reshape(meta['shape'][1:])
            out.append(t.clone())  # 持有独立内存, 避免野指针
        return out[0],out[1]

DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16}
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
            buf=f.read(o1-o0)
            out[k]=torch.frombuffer(buf,dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape']).clone()
    return out

# ═══ KV/状态缓存 ═══
class KVC:
    def __init__(self):self.kv={};self.cv={};self.rc={}
    def gkv(self,l):return self.kv.get(l)
    def gcv(self,l):return self.cv.get(l)
    def grc(self,l):return self.rc.get(l)
    def skv(self,l,k,v):self.kv[l]=(k,v)
    def ucv(self,l,s):self.cv[l]=s
    def urc(self,l,s):self.rc[l]=s
    def trunc(self,win):
        for l in list(self.kv.keys()):
            K,V=self.kv[l]
            if K.shape[2]>win:self.kv[l]=(K[:,:,-win:].contiguous(),V[:,:,-win:].contiguous())

cache=None;cur_pos=0
copy_stream=None;xfer_stream=None
_pref={};_pref_lid=-1
def prefetch_layer(lid):
    global _pref,_pref_lid
    with torch.cuda.stream(copy_stream):
        _pref={k:v.to(DEV,non_blocking=True) for k,v in layer_ram[lid].items()}
    _pref_lid=lid
def get_layer(lid):
    torch.cuda.current_stream().wait_stream(copy_stream)
    if _pref_lid!=lid:prefetch_layer(lid)
    return _pref

# ═══ 专家缓存 ═══
from concurrent.futures import ThreadPoolExecutor
pool=None
from collections import OrderedDict
exp_cache=OrderedDict();exp_lock=threading.Lock()
EXP_CACHE_MAX=540   # 540×25MB≈13.5GB, 总内存~27GB, 留5GB给系统
def get_expert(lid,eid):
    key=(lid,eid)
    with exp_lock:
        hit=exp_cache.get(key)
        if hit is not None:
            exp_cache.move_to_end(key);return hit
    g,d=rd.read_expert(lid,eid)
    with exp_lock:
        exp_cache[key]=(g,d)
        while len(exp_cache)>EXP_CACHE_MAX:exp_cache.popitem(last=False)
    return g,d

# ═══ Attention ═══
def attn(h,gpu,lid,bt,pre,cch):
    global cur_pos
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
        pids=torch.arange(cur_pos-S,cur_pos,device=DEV) if pre else torch.tensor([cur_pos],device=DEV)
        c=rope_cos[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        s=rope_sin[pids].view(1,1,S,rope_dim).expand(-1,Hq,-1,-1)
        x_rot=qa[:,:,:,:rope_dim]
        x1,x2=x_rot[...,:rope_dim//2],x_rot[...,rope_dim//2:]
        rotated=torch.cat([x1*c[...,:rope_dim//2]-x2*s[...,:rope_dim//2],
                           x2*c[...,rope_dim//2:]+x1*s[...,rope_dim//2:]],dim=-1)
        qa=torch.cat([rotated,qa[:,:,:,rope_dim:]],dim=-1)
        Kr=K[:,:,:,:rope_dim]
        K1,K2=Kr[...,:rope_dim//2],Kr[...,rope_dim//2:]
        K_rot=torch.cat([K1*c[...,:rope_dim//2]-K2*s[...,:rope_dim//2],
                         K2*c[...,rope_dim//2:]+K1*s[...,rope_dim//2:]],dim=-1)
        K=torch.cat([K_rot,K[:,:,:,rope_dim:]],dim=-1)
        if not pre:
            kv=cch.gkv(lid)
            if kv is not None:K=torch.cat([kv[0],K],2);Vm=torch.cat([kv[1],Vm],2)
        cch.skv(lid,K,Vm)
        K_att,V_att=K,Vm
        if not pre and lid in mem_heads:
            mems=mem_heads[lid].read_full(h.float(),lid,MEM_TOPK,'full_attn',device=DEV)
            if mems:
                inn=gpu.get('input_layernorm.weight')
                Km_list=[];Vm_list=[]
                for body,mpos,score in mems:
                    nb=body*torch.rsqrt(body.pow(2).mean(-1,keepdim=True)+EP)
                    if inn is not None:nb=nb*(1.0+inn.float())
                    nb=nb.to(torch.bfloat16)
                    kk=(nb@k.T).view(2,256).float()
                    if kn is not None:kk=kk*torch.rsqrt(kk.pow(2).mean(-1,keepdim=True)+EP)*(1.0+kn.float())
                    kk=kk.to(torch.bfloat16)
                    vv=(nb@v.T)
                    pid=max(0,int(mpos)) if mpos>=0 else 0
                    cm=rope_cos[pid].view(1,1,1,rope_dim).expand(-1,2,-1,-1)
                    sm=rope_sin[pid].view(1,1,1,rope_dim).expand(-1,2,-1,-1)
                    kk4=kk.view(1,2,1,256)
                    kr=kk4[:,:,:,:rope_dim]
                    k1,k2=kr[...,:rope_dim//2],kr[...,rope_dim//2:]
                    krot=torch.cat([k1*cm[...,:rope_dim//2]-k2*sm[...,:rope_dim//2],
                                    k2*cm[...,rope_dim//2:]+k1*sm[...,rope_dim//2:]],dim=-1)
                    kk=torch.cat([krot,kk4[:,:,:,rope_dim:]],dim=-1)
                    Km_list.append(kk);Vm_list.append(vv.view(1,2,1,256))
                K_att=torch.cat([torch.cat(Km_list,2).repeat_interleave(16,1),K],2)
                V_att=torch.cat([torch.cat(Vm_list,2).repeat_interleave(16,1),Vm],2)
        att=F.scaled_dot_product_attention(qa,K_att,V_att,is_causal=pre)
        gated=torch.sigmoid(qg)*att
        return gated.transpose(1,2).contiguous().view(B,S,8192)@o.T
    else:
        # DeltaNet: recall memories → blend into hidden (0.05 weight)
        if not pre and lid in mem_heads:
            mems=mem_heads[lid].read_full(h.float(),lid,MEM_TOPK,'full_attn',device=DEV)
            if mems:
                recalled=sum(body for body,_,_ in mems)/max(len(mems),1)
                h=h+0.05*recalled.to(torch.bfloat16).view(1,1,H)
        qkv=w('linear_attn.in_proj_qkv.weight');z=w('linear_attn.in_proj_z.weight');a=w('linear_attn.in_proj_a.weight');b=w('linear_attn.in_proj_b.weight');o=w('linear_attn.out_proj.weight');cv=w('linear_attn.conv1d.weight');dt=w('linear_attn.dt_bias');al=w('linear_attn.A_log');nr=w('linear_attn.norm.weight')
        if qkv is None:return h*0.99
        KD=NK*HK;VD=NV*HV;m=(h@qkv.T).transpose(1,2)
        if cv is not None:
            cw=cv
            if cw.dim()==2:cw=cw.unsqueeze(1)
            if not pre:
                cs=cch.gcv(lid)
                if cs is not None:
                    mcat=torch.cat([cs,m],-1)
                    cch.ucv(lid,mcat[:,:,-3:].clone())
                    m=F.silu(F.conv1d(mcat,cw,padding=0,groups=12288))
                else:
                    cch.ucv(lid,F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone())
                    m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
            else:
                cch.ucv(lid,F.pad(m,(3-m.shape[-1],0))[:,:,-3:].clone())
                m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,NK,HK);K=m[:,:,KD:2*KD].reshape(B,S,NK,HK);Vm=m[:,:,2*KD:].reshape(B,S,NV,HV)
        beta=(h@b.T).float().sigmoid() if b is not None else torch.ones(B,S,NV,device=DEV)
        av=(h@a.T).float() if a is not None else torch.zeros(B,S,NV,device=DEV)
        g=-al.float().exp()*F.softplus(av+dt.float())
        R=NV//NK;Q=Q.repeat_interleave(R,2);K=K.repeat_interleave(R,2)
        init=None if pre else cch.grc(lid)
        if S==1:
            ca,ls=torch_recurrent_gated_delta_rule(Q,K,Vm,g,beta,init,True,True)
        else:
            ca,ls=torch_chunk_gated_delta_rule(Q,K,Vm,g,beta,64,init,True,True)
        cch.urc(lid,ls)
        at=ca.float().contiguous().view(B,S,VD)
        if nr is not None:
            ni=at.reshape(B,S,NV,HV)
            at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.float().view(1,1,1,-1)).reshape(B,S,VD)
        zz=(h@z.T).float() if z is not None else at
        at=(at*F.silu(zz)).to(torch.bfloat16)
        return (at@o.T) if o is not None else at[:,:,:H]

def moe_forward(flat,gpu,lid):
    rw=gpu.get('mlp.gate.weight')
    probs=torch.softmax(flat.float()@rw.float().T,-1)
    tw,ti=torch.topk(probs,8,-1);tw=tw/tw.sum(-1,keepdim=True)
    uids=ti.unique().tolist()
    out=torch.zeros(flat.shape[0],H,device=DEV,dtype=torch.float32)
    futs={eid:pool.submit(get_expert,lid,eid) for eid in uids}
    gs=[];ds=[];oks=[]
    for eid in uids:
        try:gn,dn=futs[eid].result()
        except:continue
        gs.append(gn);ds.append(dn);oks.append(eid)
    E=len(oks)
    if E>0:
        if flat.shape[0]==1:
            G=torch.stack(gs)                  # [E,2048,3072] gate_up
            D=torch.stack(ds).transpose(1,2)   # [E,1024,3072] down转置
            with torch.cuda.stream(xfer_stream):
                Gd=G.to(DEV,non_blocking=True);Dd=D.to(DEV,non_blocking=True)
        else:
            # prefill: 逐专家处理 (正确但慢, 保持原始逻辑)
            Gd=[g.to(DEV,non_blocking=True) for g in gs]
            Dd=[d.to(DEV,non_blocking=True) for d in ds]
    sgw=gpu.get('mlp.shared_expert_gate.weight')
    if sgw is not None:
        sg=gpu.get('mlp.shared_expert.gate_proj.weight');su=gpu.get('mlp.shared_expert.up_proj.weight');sd=gpu.get('mlp.shared_expert.down_proj.weight')
        if sg is not None:
            gv=torch.sigmoid(flat.float()@sgw.float().T)
            out=out+gv*((F.silu(flat@sg.T)*(flat@su.T))@sd.T).float()
    if E>0:
        if flat.shape[0]==1:
            # decode批量: gate_up[E,2048,3072] (d=2048输出,h=3072输入), down[E,1024,3072]转置
            torch.cuda.current_stream().wait_stream(xfer_stream)
            gateup=torch.einsum('th,edh->ted',flat,Gd)          # [T,E,2048]
            inter=F.silu(gateup[:,:,:1024])*gateup[:,:,1024:]   # [T,E,1024]
            eout=torch.einsum('tei,eio->teo',inter,Dd)          # [T,E,3072]
            wvec=torch.stack([(tw[0][ti[0]==eid]).sum() for eid in oks]).to(DEV)
            out=out+(eout[0].float()*wvec.unsqueeze(-1)).sum(0,keepdim=True)
        else:
            # prefill逐专家: flat[T,3072], g[2048,3072]行=输出, d[3072,1024]
            torch.cuda.current_stream().wait_stream(xfer_stream)
            for j,eid in enumerate(oks):
                mask=(ti==eid).any(-1)
                if mask.any():
                    t2=flat[mask];g=Gd[j];d=Dd[j]
                    gw=(tw[mask].float()*(ti[mask]==eid)).sum(-1)
                    eout=((F.silu(t2@g[:1024].T)*(t2@g[1024:].T))@d.T).float()
                    out[mask]+=gw.unsqueeze(-1)*eout
    return out

def flush_memories():
    global cache
    wrote=0
    for lid,mh in mem_heads.items():
        n=len(pend_h[lid])
        if n<=WINDOW:continue
        old_n=n-WINDOW
        olds=torch.stack(pend_h[lid][:old_n]).float()
        positions=pend_pos[lid][:old_n]
        for s0 in range(0,old_n,FLUSH_SEG):
            seg=olds[s0:s0+FLUSH_SEG].mean(0,keepdim=True)
            segpos=positions[s0:s0+FLUSH_SEG]
            mh.write_full(seg,lid,'full_attn',pos=float(sum(segpos)/len(segpos)))
            wrote+=1
        pend_h[lid]=pend_h[lid][old_n:]
        pend_pos[lid]=pend_pos[lid][old_n:]
    cache.trunc(WINDOW)
    return wrote

# ═══ 引擎 ═══
def init_engine(log=print):
    global tk,cf,H,V,LV,EP,layer_ram,emb,lm_head,final_norm,rd,rope_cos,rope_sin,rope_dim
    global cache,cur_pos,copy_stream,xfer_stream,pool,ENGINE_READY,STATS
    t0=time.perf_counter()
    tk=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    try:
        cf=AutoConfig.from_pretrained(MODEL,trust_remote_code=True).text_config
    except Exception:
        import types as _ty
        cf=_ty.SimpleNamespace(**json.load(open(MODEL+'/config.json',encoding='utf-8'))['text_config'])  # 4.57.6不认识qwen3_5_moe, 直读json
    H=cf.hidden_size;V=cf.vocab_size;LV=cf.layer_types;EP=cf.rms_norm_eps
    log(f'[init] config ok ({cf.num_hidden_layers} layers)')
    idx=json.load(open(MODEL+'/model.safetensors.index.json'))
    layer_ram=[{} for _ in range(48)]
    shard_keys={}
    for name,shard in idx['weight_map'].items():
        if '.mlp.experts.' in name or name.startswith('mtp.') or name.startswith('model.visual'):continue
        shard_keys.setdefault(shard,[]).append(name)
    for shard,names in shard_keys.items():
        tensors=load_keys_pread(MODEL+'/'+shard,names)
        for k,v in tensors.items():
            name=k.replace('model.language_model.','')
            if 'embed_tokens' in name:emb=v
            elif 'lm_head' in name:lm_head=v
            elif name=='norm.weight':final_norm=v
            elif 'layers.' in name:
                p=name.split('.');lid=int(p[p.index('layers')+1]);key='.'.join(p[p.index('layers')+2:])
                layer_ram[lid][key]=v
        del tensors
    emb=emb.to(DEV);lm_head=lm_head.to(DEV);final_norm=final_norm.float().to(DEV)
    log(f'[init] weights loaded {time.perf_counter()-t0:.0f}s')
    rd=PreadExpertReader(MODEL)
    import os.path
    for _s in ('model.safetensors-00002-of-00039.safetensors','model.safetensors-00005-of-00039.safetensors',
               'model.safetensors-00008-of-00039.safetensors','model.safetensors-00011-of-00039.safetensors',
               'model.safetensors-00014-of-00039.safetensors','model.safetensors-00017-of-00039.safetensors',
               'model.safetensors-00026-of-00039.safetensors','model.safetensors-00029-of-00039.safetensors',
               'model.safetensors-00032-of-00039.safetensors'):
        _p='C:/ai_experts/'+_s
        if os.path.exists(_p):rd.remap(_s,_p)
    # Memory heads: auto raw/proj per layer
    MIN_SAMPLES_FOR_PROJ = 5000
    proj_dir = 'D:/ai/memory_122B/projectors'
    raw_count = 0; proj_count = 0
    for lid in range(48):
        W_path = f'{proj_dir}/L{lid:02d}_W.npy'
        b_path = f'{proj_dir}/L{lid:02d}_b.npy'
        # Count .mem files for this layer
        layer_dir = f'D:/ai/memory_122B/L{lid:02d}'
        n_mem = sum(1 for r,_,fs in os.walk(layer_dir) for f in fs if f.endswith('.mem')) if os.path.exists(layer_dir) else 0
        use_proj = os.path.exists(W_path) and os.path.exists(b_path) and n_mem >= MIN_SAMPLES_FOR_PROJ
        mh = MemoryHead(layer_dir, max_gpu=200, max_ram=2000, hidden_dim=H, mem_dim=256, raw_mode=not use_proj)
        if use_proj:
            mh.load_trained_weights(W_path, b_path)
            proj_count += 1
        else:
            raw_count += 1
        mem_heads[lid] = mh
        pend_h[lid] = []; pend_pos[lid] = []
    log(f'[init] Memory: {proj_count} layers proj, {raw_count} raw (threshold={MIN_SAMPLES_FOR_PROJ})')
    rope_dim=int(256*0.25)
    freq=1.0/(10000000**(torch.arange(0,rope_dim,2,device=DEV).float()/rope_dim))
    MAXP=8192
    rope_cos=torch.zeros(MAXP,rope_dim,device=DEV,dtype=torch.bfloat16)
    rope_sin=torch.zeros(MAXP,rope_dim,device=DEV,dtype=torch.bfloat16)
    for p in range(MAXP):
        theta=p*freq
        rope_cos[p]=torch.cat([theta.cos(),theta.cos()]).to(torch.bfloat16)
        rope_sin[p]=torch.cat([theta.sin(),theta.sin()]).to(torch.bfloat16)
    cache=KVC();cur_pos=0
    copy_stream=torch.cuda.Stream();xfer_stream=torch.cuda.Stream()
    pool=ThreadPoolExecutor(max_workers=16)
    prefetch_layer(0)
    ENGINE_READY=True
    STATS['started']=time.strftime('%H:%M:%S')
    log(f'[init] ENGINE READY in {time.perf_counter()-t0:.0f}s, open http://127.0.0.1:{PORT}')

def reset_conversation():
    global cache,cur_pos
    cache=KVC();cur_pos=0
    for lid in mem_heads:pend_h[lid]=[];pend_pos[lid]=[]
    prefetch_layer(0)

SYSTEM_PROMPT='你是暄依嫣，AI助手。当需要生成生图提示词时，必须用英文输出（FLUX只支持英文）。'

def generate_turn(msg=None,max_new=MAX_NEW_DEFAULT,raw=False,messages=None):
    """raw格式手动拼接 用户:...助手:..."""
    global cur_pos,STATS
    # ═══ ChatML + 空think壳 (模型直接回答, EOS自然停) ═══
    def build_prompt(msgs):
        parts=[]
        for m in msgs:
            role=m['role']
            content=m['content']
            if role=='system': parts.append(f'<|im_start|>system\n{content}<|im_end|>\n')
            elif role=='user': parts.append(f'<|im_start|>user\n{content}<|im_end|>\n')
            elif role=='assistant': parts.append(f'<|im_start|>assistant\n{content}<|im_end|>\n')
        parts.append('<|im_start|>assistant\n<think>\n\n</think>\n\n')  # 空think壳: 跳过思考直接回答
        return ''.join(parts)
    if messages is not None:
        prompt=build_prompt(messages)
        ids=torch.tensor([tk.encode(prompt,add_special_tokens=True)],device=DEV)
    elif raw:
        ids=torch.tensor([tk.encode(msg,add_special_tokens=True)],device=DEV)
    else:
        prompt=build_prompt([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':msg}])
        ids=torch.tensor([tk.encode(prompt,add_special_tokens=True)],device=DEV)
    S_new=ids.shape[1];p_start=cur_pos
    if cur_pos+S_new>=8000:  # 只看实际上下文长度, 不看max_new
        reset_conversation()
        yield ('info','上下文重置')
        if messages:
            last_user=[m for m in messages if m['role']=='user']
            prompt=build_prompt(last_user[-1:] if last_user else messages)
            ids=torch.tensor([tk.encode(prompt,add_special_tokens=True)],device=DEV)
        elif raw:
            ids=torch.tensor([tk.encode(msg,add_special_tokens=True)],device=DEV)
        else:
            prompt=build_prompt([{'role':'user','content':msg}])
            ids=torch.tensor([tk.encode(prompt,add_special_tokens=True)],device=DEV)
        S_new=ids.shape[1];p_start=cur_pos
    h=F.embedding(ids,emb)
    cur_pos+=S_new
    t0=time.perf_counter()
    for lid in range(48):
        bt=LV[lid]
        gpu=get_layer(lid)
        if lid+1<48:prefetch_layer(lid+1)
        else:prefetch_layer(0)
        inn=gpu.get('input_layernorm.weight')
        hn=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None:hn=hn*(1.0+inn.float())
        hn=hn.to(torch.bfloat16)
        pend_h[lid].extend(hn[0].float().cpu().unbind(0))
        pend_pos[lid].extend(range(p_start,p_start+S_new))
        a_out=attn(hn,gpu,lid,bt,True,cache)
        h=h+a_out
        pn=gpu.get('post_attention_layernorm.weight')
        hn2=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None:hn2=hn2*(1.0+pn.float())
        flat=hn2.to(torch.bfloat16).reshape(-1,H)
        out=moe_forward(flat,gpu,lid)
        h=(h+out.to(torch.bfloat16).reshape_as(h))
        del gpu
    wrote=flush_memories()
    if wrote:yield('info',f'{wrote}mem flushed')
    yield('info',f'prefill {S_new}tok {time.perf_counter()-t0:.1f}s')
    gen=[]
    # ═══ Top-p 采样 + 重复惩罚 (替代argmax, 防死循环) ═══
    TEMP=0.7; TOP_P=0.9; REP_PEN=1.12; REP_WIN=64
    gen_ids=[]
    def sample():
        hf=h.float()
        hnorm=hf*torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+EP)*(1.0+final_norm)
        logits=(hnorm[:,-1:,:].to(torch.bfloat16)@lm_head.T).float()[0,-1]
        # 重复惩罚
        if gen_ids:
            recent=set(gen_ids[-REP_WIN:])
            for tid in recent:
                if logits[tid]>0: logits[tid]/=REP_PEN
                else: logits[tid]*=REP_PEN
        logits=logits/TEMP
        probs=torch.softmax(logits,dim=-1)
        sorted_p,sorted_i=torch.sort(probs,descending=True)
        cum=torch.cumsum(sorted_p,dim=0)
        sorted_p[cum-sorted_p>TOP_P]=0
        sorted_p=sorted_p/sorted_p.sum()
        return int(sorted_i[torch.multinomial(sorted_p,1).item()])
    nt=sample()
    torch.cuda.synchronize()
    full=''
    # ═══ 状态机: <think>...</think> 标签 → 思考块; 其余 → 答案 ═══
    # THINK_MAX: 思考超过此token数强制结束进入回答 (防无限思考)
    THINK_MAX = 2048
    import re as _re
    in_think=False; think_buf=''; ans_started=False; think_len=0
    for step in range(max_new):
        gen.append(nt);gen_ids.append(nt)
        piece=tk.decode([nt],skip_special_tokens=False)
        if nt==tk.eos_token_id:break
        full+=piece
        if not ans_started:
            if not in_think:
                # 检测 <think> 开始 (可能跨token, 用全文找)
                ti=full.find('<think>')
                if ti>=0:
                    in_think=True
                    pre=full[:ti]
                    if pre: yield('tok',pre)
                    think_buf=full[ti+len('<think>'):]
                    think_len=len(think_buf)
                    # 若 <think>...</think> 空壳在同段内
                    ci=think_buf.find('</think>')
                    if ci>=0:
                        content=think_buf[:ci]
                        in_think=False; ans_started=True; think_buf=''
                        if content.strip(): yield('think',content)
                        after=full[ti+len('<think>')+ci+len('</think>'):]
                        if after.strip(): yield('tok',after)
                else:
                    yield('tok',piece)
            else:
                think_buf+=piece; think_len+=1
                ci=think_buf.find('</think>')
                if ci>=0:
                    content=think_buf[:ci]
                    after=think_buf[ci+len('</think>'):]
                    in_think=False; ans_started=True; think_buf=''
                    if content.strip(): yield('think',content)
                    if after.strip(): yield('tok',after)
                elif think_len>=THINK_MAX:
                    # 思考过长, 强制结束进入回答
                    in_think=False; ans_started=True
                    if think_buf.strip(): yield('think',think_buf)
                    think_buf=''
                    yield('info',f'[思考达上限{THINK_MAX},强制结束]')
                else:
                    yield('think',piece)
        else:
            yield('tok',piece)
        # 重复死循环熔断 (更灵敏: 连续4 token重复)
        if len(gen_ids)>=8 and gen_ids[-4:]==gen_ids[-8:-4]:
            yield('info','[熔断] 检测到重复循环,已停止')
            break
        t1=time.perf_counter()
        h=F.embedding(torch.tensor([[nt]],device=DEV),emb)
        cur_pos+=1
        for lid in range(48):
            bt=LV[lid]
            gpu=get_layer(lid)
            if lid+1<48:prefetch_layer(lid+1)
            else:prefetch_layer(0)
            inn=gpu.get('input_layernorm.weight')
            hn=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
            if inn is not None:hn=hn*(1.0+inn.float())
            hn=hn.to(torch.bfloat16)
            a_out=attn(hn,gpu,lid,bt,False,cache)
            h=h+a_out
            pn=gpu.get('post_attention_layernorm.weight')
            hn2=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
            if pn is not None:hn2=hn2*(1.0+pn.float())
            flat=hn2.to(torch.bfloat16).reshape(-1,H)
            out=moe_forward(flat,gpu,lid)
            h=(h+out.to(torch.bfloat16).reshape_as(h))
            del gpu
        nt=sample()
        torch.cuda.synchronize()
        dt=time.perf_counter()-t1
        if step==0:yield('info',f'{dt:.1f}s/token')
    STATS['tokens']+=len(gen);STATS['turns']+=1
    yield('done',full)


def generate_sync(msg, max_new=256):
    """阻塞生成完整文本(供 stage_text 等调用)"""
    if not ENGINE_READY:init_engine()
    out=''
    msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':msg}]
    for typ,data in generate_turn(messages=msgs,max_new=max_new):
        if typ=='tok':out+=data
    return out

def shutdown():
    """释放引擎内存, 把资源让给后续阶段"""
    global layer_ram,emb,lm_head,final_norm,rd,rope_cos,rope_sin,cache,ENGINE_READY
    import gc
    layer_ram=None;emb=None;lm_head=None;final_norm=None;rd=None
    rope_cos=None;rope_sin=None;cache=None;ENGINE_READY=False
    mem_heads.clear()
    gc.collect();torch.cuda.empty_cache()

# ═══ HTTP 服务 ═══
INDEX_HTML=None
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*a):pass
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type,Authorization')
    def do_OPTIONS(self):
        self.send_response(204);self._cors();self.end_headers()
    def _openai_chat(self):
        ln=int(self.headers.get('Content-Length',0))
        req=json.loads(self.rfile.read(ln).decode('utf-8',errors='replace'))
        messages=req.get('messages',[])
        msgs=[{'role':'system','content':SYSTEM_PROMPT}]+messages[-8:]
        stream=req.get('stream',False)
        max_new=int(req.get('max_tokens',MAX_NEW_DEFAULT))  # 无上限, EOS自然停
        if not ENGINE_READY:
            body=json.dumps({'error':'loading'}).encode()
            self.send_response(503);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self._cors();self.end_headers()
            self.wfile.write(body);return
        if stream:
            self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
            self.send_header('Cache-Control','no-cache');self._cors();self.end_headers()
            with LOCK:
                try:
                    full='';reason=''
                    for typ,data in generate_turn(messages=msgs,max_new=max_new):
                        if typ=='tok':
                            full+=data
                            chunk={'id':f'chatcmpl-{time.time()}','object':'chat.completion.chunk','created':int(time.time()),'model':'qwen3.5-122b','choices':[{'index':0,'delta':{'content':data},'finish_reason':None}]}
                            self.wfile.write(f'data: {json.dumps(chunk,ensure_ascii=False)}\n\n'.encode());self.wfile.flush()
                        elif typ=='think':
                            reason+=data
                            chunk={'id':f'chatcmpl-{time.time()}','object':'chat.completion.chunk','created':int(time.time()),'model':'qwen3.5-122b','choices':[{'index':0,'delta':{'reasoning_content':data},'finish_reason':None}]}
                            self.wfile.write(f'data: {json.dumps(chunk,ensure_ascii=False)}\n\n'.encode());self.wfile.flush()
                        elif typ=='done':
                            final={'id':f'chatcmpl-{time.time()}','object':'chat.completion.chunk','created':int(time.time()),'model':'qwen3.5-122b','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':0,'completion_tokens':len(full),'total_tokens':len(full)}}
                            self.wfile.write(f'data: {json.dumps(final,ensure_ascii=False)}\n\n'.encode());self.wfile.write('data: [DONE]\n\n'.encode());self.wfile.flush()
                except Exception as e:
                    import traceback;traceback.print_exc()
        else:
            with LOCK:
                out='';reason=''
                for typ,data in generate_turn(messages=msgs,max_new=max_new):
                    if typ=='tok':out+=data
                    elif typ=='think':reason+=data
            body=json.dumps({'id':f'chatcmpl-{time.time()}','object':'chat.completion','created':int(time.time()),'model':'qwen3.5-122b','choices':[{'index':0,'message':{'role':'assistant','content':out,'reasoning_content':reason},'finish_reason':'stop'}],'usage':{'prompt_tokens':0,'completion_tokens':len(out),'total_tokens':len(out)}},ensure_ascii=False).encode()
            body=json.dumps({'id':f'chatcmpl-{time.time()}','object':'chat.completion','created':int(time.time()),'model':'qwen3.5-122b','choices':[{'index':0,'message':{'role':'assistant','content':out},'finish_reason':'stop'}],'usage':{'prompt_tokens':0,'completion_tokens':len(out),'total_tokens':len(out)}},ensure_ascii=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self._cors();self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/images/'):
            fp=r'D:\ai\output'+self.path
            if os.path.exists(fp):
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(os.path.getsize(fp)));self._cors();self.end_headers()
                with open(fp,'rb') as f:self.wfile.write(f.read())
            else:
                self.send_response(404);self.end_headers()
        elif self.path=='/' or self.path=='/index.html':
            body=open(r'D:\ai\frontend\index.html','rb').read()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self._cors();self.end_headers();self.wfile.write(body)
        elif self.path=='/v1/models':
            body=json.dumps({'object':'list','data':[{'id':'qwen3.5-122b','object':'model','created':1728000000,'owned_by':'local'}]},ensure_ascii=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self._cors();self.end_headers();self.wfile.write(body)
        elif self.path=='/api/status':
            mem_total=sum(mh.writes for mh in mem_heads.values())
            data=json.dumps({'ready':ENGINE_READY,'tokens':STATS['tokens'],'turns':STATS['turns'],'started':STATS['started'],'memories':mem_total,'kv_len':(cache.kv.get(3,(' ',' '))[0].shape[2] if cache and 3 in cache.kv else 0)},ensure_ascii=False).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self._cors();self.end_headers();self.wfile.write(data)
        else:
            self.send_response(404);self.end_headers()

    def _send_json(self,data,code=200):
        body=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self._cors();self.end_headers();self.wfile.write(body)

    def do_POST(self):
        if self.path=='/v1/chat/completions':
            self._openai_chat();return
        if self.path=='/api/reset':
            with LOCK:
                if ENGINE_READY:reset_conversation()
            self._send_json({'ok':True});return
        if self.path=='/api/generate':
            # 生图: 停122B→FLUX→4图→返回路径(FLUX保持运行)
            ln=int(self.headers.get('Content-Length',0))
            try:req=json.loads(self.rfile.read(ln).decode('utf-8',errors='replace'))
            except:req={}
            prompt=(req.get('prompt') or '').strip()
            steps=min(int(req.get('steps',50)),200)  # 2K高清: 默认50步(官方推荐上限), 最高200
            count=min(int(req.get('count',4)),30)    # 前端可调张数, 上限30 (防刷超时)
            if not prompt:
                self._send_json({'error':'no prompt'},400);return
            print(f'[gen] prompt: {prompt[:80]} steps={steps} count={count}',flush=True)
            with LOCK:
                if ENGINE_READY:
                    shutdown()
                    print('[gen] 122B off, FLUX on',flush=True)
            import threading as _th
            result_holder={}
            def _run():
                try:
                    import gen_image as gi
                    paths=gi.generate_batch(prompt,steps=steps,count=count,size=2560,height=1408)
                    result_holder['paths']=paths
                    print(f'[gen] done: {paths}',flush=True)
                except Exception as e:
                    import traceback;traceback.print_exc()
                    result_holder['error']=str(e)
            # 同步等待 (生图 ~2-4min), 简化前端逻辑
            _run()
            if 'error' in result_holder:
                self._send_json({'error':result_holder['error']},500)
            else:
                self._send_json({'images':result_holder['paths'],'prompt':prompt})
            return
        if self.path=='/api/return_chat':
            # 停止FLUX, 重启122B
            import gen_image as gi
            gi.free_pipe()
            print('[gen] FLUX off, restarting 122B...',flush=True)
            import threading as _th
            _th.Thread(target=init_engine,daemon=True).start()
            self._send_json({'ok':True,'msg':'122B重启中'})
            return
        if self.path!='/api/chat':
            self.send_response(404);self.end_headers();return
        ln=int(self.headers.get('Content-Length',0))
        try:req=json.loads(self.rfile.read(ln).decode('utf-8',errors='replace'))
        except:req={}
        msg=(req.get('message') or '').strip()
        max_new=int(req.get('max_new',MAX_NEW_DEFAULT))  # 无上限, 模型EOS/熔断自然停
        if not msg:
            body=json.dumps({'error':'empty'}).encode();self.send_response(400)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self._cors();self.end_headers();self.wfile.write(body);return
        self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8')
        self.send_header('Cache-Control','no-cache');self.send_header('Connection','close')
        self._cors();self.end_headers()
        def send(obj):
            self.wfile.write(('data: '+json.dumps(obj,ensure_ascii=False)+'\n\n').encode('utf-8'));self.wfile.flush()
        msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':msg}]
        with LOCK:
            try:
                for typ,data in generate_turn(messages=msgs,max_new=max_new):
                    if typ=='tok':send({'tok':data})
                    elif typ=='think':send({'think':data})
                    elif typ=='info':send({'info':data})
                    elif typ=='done':send({'done':True,'text':data})
            except Exception as e:
                import traceback;send({'error':str(e)});traceback.print_exc()

class ThreadedServer(socketserver.ThreadingMixIn,http.server.HTTPServer):
    daemon_threads=True
    allow_reuse_address=True

def main():
    import threading as th
    def _safe_init():
        print('[init] Thread started...',flush=True)
        try:init_engine()
        except Exception as e:
            import traceback;traceback.print_exc()
            print(f'[FATAL] Engine init failed: {e}',flush=True)
    t=th.Thread(target=_safe_init,daemon=True)
    t.start()
    srv=ThreadedServer(('127.0.0.1',PORT),Handler)
    sys.stdout.flush()
    print(f'[server] http://127.0.0.1:{PORT} (引擎后台加载中...)',flush=True)
    try:srv.serve_forever()
    except KeyboardInterrupt:pass

if __name__=='__main__':
    main()
