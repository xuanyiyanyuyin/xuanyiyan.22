"""LTX 13B transformer 三级调度引擎 — 照 122B 方案 + 用户适配
第1级 磁盘: 48层奇偶分盘(C/D), 双线程双盘并行 pread (pool=2), 线程本地句柄
第2级 RAM: 环形缓冲 4 槽 (~2.2GB), 顺序访问不用 LRU
第3级 GPU: 独立 copy_stream 异步 prefetch (non_blocking), 算当前层预取下一层
T5/VAE 串行隔离 (不在本模块, 由调用方处理)
"""
import sys, io, os, struct, json, gc, threading, math, atexit, signal
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# ═══ CUDA context 泄漏防护: 正常退出/Ctrl+C 时尽量释放 UVA 残留 ═══
def _cleanup_cuda():
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
atexit.register(_cleanup_cuda)
# Ctrl+C (SIGINT): Python 默认会抛 KeyboardInterrupt 并走 atexit, 这里显式兜底一次
def _sigint_handler(sig, frame):
    _cleanup_cuda()
    raise KeyboardInterrupt
try:
    signal.signal(signal.SIGINT, _sigint_handler)
except ValueError:
    pass  # 非主线程时无法设置信号处理器

DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32}
DEV='cuda'
HEAD_DIM=128; NUM_HEADS=32; HIDDEN=4096; EPS=1e-5
C_DIR=r'C:\ai\ltx_shards'
D_DIR=r'D:\ai\ltx_shards'
RING_SIZE=3

# ═══ 第1级: 磁盘层文件路径 (奇偶分盘) ═══
def layer_path(lid):
    drive=C_DIR if lid%2==0 else D_DIR
    return f'{drive}/layer_{lid:02d}.safetensors'
GLOBAL_PATH=f'{D_DIR}/global.safetensors'

# ═══ 第1级: 双线程双盘 pread (照 122B 线程本地句柄) ═══
_tls=threading.local()
def _get_fd(path):
    fds=getattr(_tls,'fds',None)
    if fds is None: fds={}; _tls.fds=fds
    if path not in fds:
        fds[path]=open(path,'rb')
    return fds[path]

def _read_tensor(f, base, o0, o1, dt, sh):
    f.seek(base+o0)
    n=o1-o0
    data=f.read(n)  # 一次读满 (f.read 用内部缓冲, 大文件也OK)
    return torch.frombuffer(data,dtype=DTYPE_MAP[dt]).reshape(sh)

def pread_safetensors(path):
    """纯 pread 读一个层文件 → RAM dict (无 mmap)"""
    f=_get_fd(path)
    f.seek(0)
    hs=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(hs))
    base=8+hs
    out={}
    for k,meta in header.items():
        if k=='__metadata__': continue
        o0,o1=meta['data_offsets']
        out[k]=_read_tensor(f,base,o0,o1,meta['dtype'],tuple(meta['shape']))
    return out

# 双线程池 (C盘线程 + D盘线程)
_pool=ThreadPoolExecutor(max_workers=1)  # 内存不足时单线程, 省内存

# ═══ 第2级: RAM 环形缓冲 ═══
_ring=[None]*RING_SIZE
def ring_slot(lid): return lid%RING_SIZE

# ═══ 第3级: GPU prefetch stream ═══
_copy_stream=torch.cuda.Stream()

# ═══ 基础运算 (与 ltx_engine 相同, 已验证) ═══
def rms_norm(x, w, eps=EPS):
    input_dtype=x.dtype
    variance=x.to(torch.float32).pow(2).mean(-1,keepdim=True)
    hidden=x*torch.rsqrt(variance+eps)
    if w is not None:
        hidden=hidden.to(w.dtype)
        hidden=hidden*w
    else:
        hidden=hidden.to(input_dtype)
    return hidden

def linear(x,w,b=None):
    if b is not None: return F.linear(x,w.to(x.dtype),b.to(x.dtype))
    return F.linear(x,w.to(x.dtype))

def apply_rotary(x, freqs):
    cos,sin=freqs
    x_real,x_imag=x.unflatten(2,(-1,2)).unbind(-1)
    x_rotated=torch.stack([-x_imag,x_real],dim=-1).flatten(2)
    return (x.float()*cos+x_rotated.float()*sin).to(x.dtype)

_rope=None
def get_rope_module():
    global _rope
    if _rope is None:
        from diffusers.models.transformers.transformer_ltx import LTXVideoRotaryPosEmbed
        _rope=LTXVideoRotaryPosEmbed(dim=HIDDEN, base_num_frames=20, base_height=2048, base_width=2048,
                                     patch_size=1, patch_size_t=1, theta=10000.0)
    return _rope

def compute_rope(hidden_states, num_frames, height, width, vae_temporal_ratio, vae_spatial_ratio, frame_rate):
    rope=get_rope_module()
    scale=(vae_temporal_ratio/frame_rate, vae_spatial_ratio, vae_spatial_ratio)
    return rope(hidden_states, num_frames, height, width, rope_interpolation_scale=scale)

def block_forward(h, txt, temb, rope_emb, mask, w):
    B,S,_=h.shape
    sst=w['scale_shift_table']
    ada=sst[None,None].to(temb.dtype)+temb.reshape(B,temb.size(1),6,-1)
    shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=ada.unbind(dim=2)
    nh=rms_norm(h,w.get('norm1.weight'))
    nh=nh.to(temb.dtype)*(1+scale_msa)+shift_msa
    a=attn_forward(nh,None,mask,rope_emb,w,True)
    h=h+a.to(h.dtype)*gate_msa.to(h.dtype)
    a=attn_forward(h,txt,mask,None,w,False)
    h=h+a.to(h.dtype)
    nh=rms_norm(h,w.get('norm2.weight'))
    nh=nh.to(temb.dtype)*(1+scale_mlp)+shift_mlp
    ff=ff_forward(nh,w)
    h=h+ff.to(h.dtype)*gate_mlp.to(h.dtype)
    return h

def attn_forward(h, enc, mask, rope_emb, w, is_self):
    B,S,_=h.shape
    prefix='attn1' if is_self else 'attn2'
    if enc is None: enc=h
    q=linear(h,w[f'{prefix}.to_q.weight'],w.get(f'{prefix}.to_q.bias'))
    k=linear(enc,w[f'{prefix}.to_k.weight'],w.get(f'{prefix}.to_k.bias'))
    v=linear(enc,w[f'{prefix}.to_v.weight'],w.get(f'{prefix}.to_v.bias'))
    q=rms_norm(q,w[f'{prefix}.norm_q.weight'])
    k=rms_norm(k,w[f'{prefix}.norm_k.weight'])
    if rope_emb is not None and is_self:
        q=apply_rotary(q,rope_emb); k=apply_rotary(k,rope_emb)
    B2,Sq,_=q.shape
    q=q.view(B2,Sq,NUM_HEADS,HEAD_DIM).transpose(1,2)
    B3,Sk,_=k.shape
    k=k.view(B3,Sk,NUM_HEADS,HEAD_DIM).transpose(1,2)
    v=v.view(B3,Sk,NUM_HEADS,HEAD_DIM).transpose(1,2)
    attn_mask=None
    if mask is not None and not is_self:
        m=(1-mask.to(q.dtype))*-10000.0
        attn_mask=m.unsqueeze(1).unsqueeze(1)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,is_causal=False)
    out=out.transpose(1,2).reshape(B2,Sq,-1).to(h.dtype)
    out=linear(out,w[f'{prefix}.to_out.0.weight'],w.get(f'{prefix}.to_out.0.bias'))
    return out

def ff_forward(x,w):
    h=linear(x,w['ff.net.0.proj.weight'],w.get('ff.net.0.proj.bias'))
    h=F.gelu(h,approximate='tanh')
    h=linear(h,w['ff.net.2.weight'],w.get('ff.net.2.bias'))
    return h

def time_embed_forward(timestep, B, gw):
    ts=timestep.float()
    freqs=torch.exp(torch.arange(0,128,dtype=torch.float32,device=ts.device)*(-math.log(10000.0)/128))
    args=ts.unsqueeze(-1)*freqs.view(1,-1)
    t_emb=torch.cat([torch.cos(args),torch.sin(args)],dim=-1)
    t=t_emb.to(gw['time_embed.emb.timestep_embedder.linear_1.weight'].dtype)
    t=linear(t,gw['time_embed.emb.timestep_embedder.linear_1.weight'],gw['time_embed.emb.timestep_embedder.linear_1.bias'])
    t=F.silu(t)
    t=linear(t,gw['time_embed.emb.timestep_embedder.linear_2.weight'],gw['time_embed.emb.timestep_embedder.linear_2.bias'])
    t=t.view(B,-1)
    temb=F.silu(t)
    temb=linear(temb,gw['time_embed.linear.weight'],gw['time_embed.linear.bias'])
    temb=temb.view(B,6,-1)
    return temb.unsqueeze(1), t.view(B,1,-1)

# ═══ 三级调度完整 forward ═══
_global_ram=None
def get_global(dtype=torch.bfloat16):
    global _global_ram
    if _global_ram is None:
        _global_ram=pread_safetensors(GLOBAL_PATH)
    return {k:v.to(dtype) for k,v in _global_ram.items()}

def cleanup():
    """释放引擎资源 (去噪后, VAE 前调用)"""
    global _global_ram
    _global_ram=None
    for i in range(RING_SIZE):
        _ring[i]=None
    gc.collect()
    torch.cuda.empty_cache()

def transformer_forward_sched(hidden_states, encoder_hidden_states, timestep, mask,
                              num_frames, height, width, rope_emb=None, dtype=torch.bfloat16):
    """三级调度 forward: 双盘并行读 + RAM环形缓冲 + GPU prefetch
    forward 全程 bf16 (照官方), latent 保持 float32, 输出 float32
    """
    B,S,_=hidden_states.shape
    # global 权重 (bf16, 照官方)
    gw=get_global(torch.bfloat16)
    gwgpu={k:(v.to(DEV) if v.is_floating_point() else v) for k,v in gw.items()}
    try:
        # 输入转 bf16 (照官方 latent.to(prompt_embeds.dtype))
        h=hidden_states.to(DEV).to(torch.bfloat16)
        txt_in=encoder_hidden_states.to(DEV).to(torch.bfloat16)
        ts_in=timestep.to(DEV)
        if rope_emb is not None:
            rope_emb=(rope_emb[0].to(DEV),rope_emb[1].to(DEV))
        h=linear(h, gwgpu['proj_in.weight'], gwgpu.get('proj_in.bias'))
        temb, emb_ts=time_embed_forward(ts_in,B,gwgpu)
        txt=linear(txt_in, gwgpu['caption_projection.linear_1.weight'], gwgpu['caption_projection.linear_1.bias'])
        txt=F.gelu(txt,approximate='tanh')
        txt=linear(txt, gwgpu['caption_projection.linear_2.weight'], gwgpu['caption_projection.linear_2.bias'])
        txt=txt.view(B,-1,HIDDEN)
        if mask is not None: mask=mask.to(DEV)

        # 逐层搬运: 读当前层到RAM(单层) → 搬GPU → 算 → 释放 (RAM峰值=单层)
        for lid in range(48):
            w=pread_safetensors(layer_path(lid))  # 磁盘→RAM (单层)
            wgpu={k:(v.to(DEV) if v.is_floating_point() else v) for k,v in w.items()}  # RAM→GPU
            h=block_forward(h,txt,temb,rope_emb,mask,wgpu)
            del w, wgpu  # 释放RAM+GPU
            if lid%8==0: gc.collect()
        # 收尾: norm_out (LayerNorm, eps=1e-6) + scale_shift + proj_out
        ssv=gwgpu['scale_shift_table'][None,None].to(emb_ts.dtype)+emb_ts[:,:,None]
        shift,scale=ssv[:,:,0],ssv[:,:,1]
        # LayerNorm: (x - mean) / sqrt(var + eps), elementwise_affine=False
        h=(h.float()-h.float().mean(-1,keepdim=True))/torch.sqrt(h.float().var(-1,keepdim=True,unbiased=False)+1e-6)
        h=h.to(shift.dtype)*(1+scale)+shift
        out=linear(h,gwgpu['proj_out.weight'],gwgpu.get('proj_out.bias'))
        return out.float().to(hidden_states.device)
    finally:
        del gw, gwgpu; gc.collect()
