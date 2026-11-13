"""LTX-2.5 权重验证 — Gemma4 / DiT 形状 + RoPE pe 探测
用法: python ltx25_verify.py gemma | dit | rope
"""
import sys, torch, json, struct

def load(path):
    import safetensors.torch
    return safetensors.torch.load_file(path, device='cpu')

def verify_gemma():
    sd = load(r'D:\ai\_ltx25_src\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors')
    print(f'Gemma4: {len(sd)} tensors')
    # 找 projection / embed / hidden
    keys = sorted(sd.keys())
    for k in keys:
        if any(x in k.lower() for x in ['projection','proj','embed','hidden','connector']):
            print(f'  {k}: {tuple(sd[k].shape)} {sd[k].dtype}')
    # 推断 hidden_size (从最大 linear 权重)
    maxdim = max((sd[k].shape for k in keys if sd[k].ndim==2), key=lambda s: max(s))
    print(f'  最大2D权重: {maxdim}')

def verify_dit():
    sd = load(r'D:\ai\_ltx25_src\ltx-2.5-22b-dev-transformer-bf16_v2.safetensors')
    print(f'DiT: {len(sd)} tensors')
    keys = sorted(sd.keys())
    # 关键模块: patchify_proj / adaln / caption_projection / attn / ff
    targets = ['patchify','adaln','caption','proj_in','proj_out','attn1','attn2','to_q','to_k','to_v','to_out','ff.','norm']
    seen = set()
    for k in keys:
        if 'layers.' in k:  # 只统计层0的
            continue
        if k.startswith(('patchify','adaln','caption','time_embed','proj_in','proj_out','norm_out','scale_shift')):
            print(f'  [顶层] {k}: {tuple(sd[k].shape)}')
    # 层0的 attn/ff
    for k in keys:
        if '.0.' in k and any(t in k for t in ['attn1','attn2','ff.','norm1','norm2']):
            if 'layers.0.' in k and k.count('.') >= 3:
                if seen.get(k.split('.')[2]+'.'+k.split('.')[3], 0) < 3:
                    print(f'  [层0] {k}: {tuple(sd[k].shape)}')
                    seen[k.split('.')[2]+'.'+k.split('.')[3]] = seen.get(k.split('.')[2]+'.'+k.split('.')[3],0)+1

def verify_rope():
    sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
    from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.rope import LTXRopeType
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle
    # 小尺寸构造
    import torch.nn as nn
    inner_dim = 4096; max_pos = [20,2048,2048]
    patchify_proj = nn.Linear(128, inner_dim)
    adaln = AdaLayerNormSingle(inner_dim)
    pre = TransformerArgsPreprocessor(
        patchify_proj=patchify_proj.to(torch.bfloat16), adaln=adaln.to(torch.bfloat16), inner_dim=inner_dim,
        max_pos=max_pos, num_attention_heads=32, use_middle_indices_grid=True,
        timestep_scale_multiplier=1000, double_precision_rope=False,
        positional_embedding_theta=10000.0, rope_type=LTXRopeType.SPLIT,
        caption_projection=nn.Linear(3840, inner_dim).to(torch.bfloat16),
    )
    T,H,W = 2,4,4  # 小 latent
    lat = torch.randn(1, T*H*W, 128, dtype=torch.bfloat16)
    pos = torch.randn(1, 3, T*H*W, 2, dtype=torch.float32)  # middle indices grid
    mod = Modality(latent=lat, sigma=torch.tensor([0.5]), timesteps=torch.tensor([500.0]),
                   positions=pos, context=torch.randn(1, 16, 3840, dtype=torch.bfloat16),
                   enabled=True)
    args = pre.prepare(mod)
    pe = args.positional_embeddings
    print(f'pe 类型: {type(pe)}')
    if isinstance(pe, tuple):
        print(f'pe tuple len={len(pe)}, 每个 shape: {[tuple(p.shape) for p in pe]} dtype: {[p.dtype for p in pe]}')
    else:
        print(f'pe shape: {tuple(pe.shape)} dtype: {pe.dtype}')
        print(f'pe 前几值: {pe.flatten()[:5]}')
    print(f'x: {tuple(args.x.shape)}, timesteps: {tuple(args.timesteps.shape)}, context: {tuple(args.context.shape)}')

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv)>1 else 'all'
    if which in ('gemma','all'): verify_gemma()
    if which in ('dit','all'): verify_dit()
    if which in ('rope','all'): verify_rope()
