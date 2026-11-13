"""LTX 13B transformer 手写推理引擎 — 完全照 122B 方案
权重: PreadLayerReader 按层从磁盘 pread 读取, 用完释放 (不常驻, 内存峰值=单层)
显存: 逐层搬 GPU, 用完释放; 长序列分块注意力
"""
import sys, io, os, struct, json, gc, threading, math
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32}
DEV='cuda'
HEAD_DIM=128; NUM_HEADS=32; HIDDEN=4096; EPS=1e-5

TRANSFORMER_RENAME = {
    'model.diffusion_model.': '',
    'patchify_proj': 'proj_in',
    'adaln_single': 'time_embed',
    'q_norm': 'norm_q',
    'k_norm': 'norm_k',
}
def convert_ltx_tf_keys(raw_key):
    k=raw_key
    for a,b in TRANSFORMER_RENAME.items():
        k=k.replace(a,b)
    return k

# ═══ PreadLayerReader: 按层从磁盘读取, 不常驻 (照 122B PreadExpertReader) ═══
class PreadLayerReader:
    def __init__(self, single_file):
        self.path=single_file
        self._tl=threading.local()
        with open(single_file,'rb') as f:
            hs=struct.unpack('<Q',f.read(8))[0]
            header=json.loads(f.read(hs))
            self.base=8+hs
        self._hdr=header
        self.layer_locs={}; self.global_locs=[]
        for raw,k_meta in header.items():
            if not raw.startswith('model.diffusion_model.'): continue
            ck=convert_ltx_tf_keys(raw)
            o0,o1=k_meta['data_offsets']; dt=k_meta['dtype']; sh=tuple(k_meta['shape'])
            if ck.startswith('transformer_blocks.'):
                lid=int(ck.split('.')[1])
                self.layer_locs.setdefault(lid,[]).append((ck,o0,o1,dt,sh))
            else:
                self.global_locs.append((ck,o0,o1,dt,sh))
        print(f'[PreadReader] layers={len(self.layer_locs)} global={len(self.global_locs)}',flush=True)

    def _fh(self):
        h=getattr(self._tl,'h',None)
        if h is None: h={}; self._tl.h=h
        return h
    def _read(self,o0,o1,dt,sh):
        fh=self._fh()
        f=fh.get(self.path)
        if f is None:
            f=open(self.path,'rb'); fh[self.path]=f
        f.seek(self.base+o0)
        buf=bytearray(f.read(o1-o0))
        return torch.frombuffer(buf,dtype=DTYPE_MAP[dt]).reshape(sh)

    def get_global(self, dtype=torch.bfloat16):
        out={}
        for ck,o0,o1,dt,sh in self.global_locs:
            t=self._read(o0,o1,dt,sh)
            out[ck]=t.to(dtype)
        return out

    def get_layer(self, lid, dtype=torch.bfloat16):
        w={}
        locs=self.layer_locs.get(lid,[])
        for ck,o0,o1,dt,sh in locs:
            t=self._read(o0,o1,dt,sh)
            short=ck
            prefix=f'transformer_blocks.{lid}.'
            if short.startswith(prefix):
                short=short[len(prefix):]
            w[short]=t.to(dtype)
        return w

# ═══ 基础运算 ═══
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
    if b is not None:
        return F.linear(x,w.to(x.dtype),b.to(x.dtype))
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

# ═══ 单层 forward ═══
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

# ═══ 完整 forward: 逐层读取, 用完释放 ═══
def transformer_forward(reader, hidden_states, encoder_hidden_states, timestep, mask,
                        num_frames, height, width, rope_emb=None, dtype=torch.bfloat16):
    B,S,_=hidden_states.shape
    gw=reader.get_global(dtype)
    gwgpu={k:(v.to(DEV) if v.is_floating_point() else v) for k,v in gw.items()}
    try:
        h=hidden_states.to(DEV)
        txt_in=encoder_hidden_states.to(DEV)
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
        for lid in sorted(reader.layer_locs.keys()):
            w=reader.get_layer(lid,dtype)
            wgpu={k:(v.to(DEV) if v.is_floating_point() else v) for k,v in w.items()}
            h=block_forward(h,txt,temb,rope_emb,mask,wgpu)
            del w, wgpu; gc.collect()
        ssv=gwgpu['scale_shift_table'][None,None].to(emb_ts.dtype)+emb_ts[:,:,None]
        shift,scale=ssv[:,:,0],ssv[:,:,1]
        h=rms_norm(h,None)
        h=h.to(shift.dtype)*(1+scale)+shift
        h=h.to(dtype)
        out=linear(h,gwgpu['proj_out.weight'],gwgpu.get('proj_out.bias'))
        return out.to(hidden_states.device)
    finally:
        del gw, gwgpu; gc.collect()
