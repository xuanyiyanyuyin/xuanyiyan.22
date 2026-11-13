"""LTX VAE 1024 pread 加载 — 照 t5_loader 绕开 from_pretrained mmap 段错误"""
import sys, io, os, struct, json, gc
sys.path.insert(0, r'D:\ai')
import torch

DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32}

def load_vae_weights(single_file):
    """pread 读取 VAE 全部权重 (无 mmap)"""
    out={}
    with open(single_file,'rb') as f:
        hs=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(hs))
        base=8+hs
    for k,meta in header.items():
        if k=='__metadata__': continue
        o0,o1=meta['data_offsets']
        with open(single_file,'rb') as f:  # 每次重新打开, 避免句柄关闭
            f.seek(base+o0)
            n=o1-o0
            buf=bytearray()
            CHUNK=64*1024*1024
            while len(buf)<n:
                c=f.read(min(CHUNK,n-len(buf)))
                if not c: break
                buf.extend(c)
        out[k]=torch.frombuffer(bytes(buf),dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
    return out

def load_vae_pread(model_dir, dtype=torch.bfloat16):
    """用 pread 加载 AutoencoderKLLTXVideo (vae_1024)
    边读边填: 先创建空模型, 逐 tensor 读权重直接填入参数 (不累积 dict)
    """
    from diffusers import AutoencoderKLLTXVideo
    import json as _json, os
    cfg_path=os.path.join(model_dir,'config.json')
    cfg=_json.load(open(cfg_path))
    vae=AutoencoderKLLTXVideo.from_config(cfg)  # 创建空模型 (参数已分配)
    # 收集参数名→参数对象
    param_map={n:p for n,p in vae.named_parameters()}
    buf_map={n:b for n,b in vae.named_buffers()}
    # 读 header
    sf=os.path.join(model_dir,'diffusion_pytorch_model.safetensors')
    with open(sf,'rb') as f:
        hs=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(hs))
        base=8+hs
    n_loaded=0
    CHUNK=16*1024*1024  # 16MB 分块读
    for k,meta in header.items():
        if k=='__metadata__': continue
        o0,o1=meta['data_offsets']
        n=o1-o0
        with open(sf,'rb') as f:
            f.seek(base+o0)
            buf=bytearray()
            while len(buf)<n:
                c=f.read(min(CHUNK,n-len(buf)))
                if not c: break
                buf.extend(c)
        t=torch.frombuffer(bytes(buf),dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
        # 填入参数/buffer
        if k in param_map:
            param_map[k].data.copy_(t.to(dtype))
            n_loaded+=1
        elif k in buf_map:
            buf_map[k].data.copy_(t.to(dtype))
            n_loaded+=1
        else:
            print(f'[vae-pread] 未匹配: {k}',flush=True)
        del t, buf
    print(f'[vae-pread] loaded {n_loaded} keys',flush=True)
    # ⚠ copy_ 不改参数 dtype: from_config 创建 f32 参数, copy_(bf16) 会把 bf16 值存进 f32。
    #   模型实际保持 f32 (值来自 bf16 checkpoint, 无损上采样)。对齐全量 f32, 调用方须喂 f32 输入。
    #   (不要转 bf16: 官方虽用 bf16, 但 f32 decode 更稳, 且值相同)
    vae = vae.to(torch.float32)
    gc.collect()
    return vae

if __name__=='__main__':
    v=load_vae_pread(r'D:\ai\models\LTX-Video\vae_1024')
    print('VAE loaded', v.config.latent_channels, flush=True)
