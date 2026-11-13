# -*- coding: utf-8 -*-
"""分步二分: 官方单块 Parallel 链, 每步 sync 后打印, 找出非法访问的真凶 kernel."""
import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2ParallelSelfAttention, Flux2ParallelSelfAttnProcessor, Flux2PosEmbed,
    apply_rotary_emb, dispatch_attention_fn,
)

DEV = 'cuda'
torch.manual_seed(1)
S = 51712
HEADS, HDIM, DIM = 48, 128, 6144

pos = Flux2PosEmbed(theta=10000, axes_dim=[32, 32, 32, 32]).to(DEV)
ids = torch.zeros(S, 4, dtype=torch.int64, device=DEV)
idx = torch.arange(S, device=DEV)
ids[:, 1] = idx % 160
ids[:, 2] = (idx // 160) % 160
cos, sin = pos(ids)

attn = Flux2ParallelSelfAttention(
    query_dim=DIM, heads=HEADS, dim_head=HDIM, out_dim=DIM,
    bias=False, out_bias=True, eps=1e-6,
    mlp_ratio=3.0, mlp_mult_factor=2,
    processor=Flux2ParallelSelfAttnProcessor(),
)
attn = attn.to(DEV).eval().bfloat16()

hidden = torch.randn(1, S, DIM, device=DEV, dtype=torch.bfloat16)

def step(name, fn):
    torch.cuda.synchronize()
    try:
        fn()
        torch.cuda.synchronize()
        print(f'[OK] {name}', flush=True)
    except Exception as e:
        print(f'[FAIL] {name}: {type(e).__name__}: {str(e)[:90]}', flush=True)
        raise SystemExit

with torch.no_grad():
    step('proj', lambda: attn.to_qkv_mlp_proj(hidden))
    h = attn.to_qkv_mlp_proj(hidden)
    step('split', lambda: torch.split(h, [3*attn.inner_dim, h.shape[-1]-3*attn.inner_dim], dim=-1))
    qkv = h
    step('chunk', lambda: qkv.chunk(3, dim=-1))
    q = qkv.chunk(3, dim=-1)[0]
    step('unflatten', lambda: q.unflatten(-1, (-1, attn.head_dim)))
    q = q.unflatten(-1, (-1, attn.head_dim))
    step('norm_q', lambda: attn.norm_q(q))
    q = attn.norm_q(q)
    step('rope_q', lambda: apply_rotary_emb(q, (cos, sin), sequence_dim=1))
    q = apply_rotary_emb(q, (cos, sin), sequence_dim=1)
    k = attn.norm_k(qkv.chunk(3, dim=-1)[1].unflatten(-1, (-1, attn.head_dim)))
    v = qkv.chunk(3, dim=-1)[2].unflatten(-1, (-1, attn.head_dim))
    step('dispatch_attn', lambda: dispatch_attention_fn(q, k, v))
print('MINI3_DONE', flush=True)
