# -*- coding: utf-8 -*-
"""S=51712 直击官方单块真实内核: Flux2ParallelSelfAttention(平行attn+mlp) + rope, 随机权重裸跑."""
import torch, time
from diffusers.models.transformers.transformer_flux2 import (
    Flux2ParallelSelfAttention, Flux2ParallelSelfAttnProcessor, Flux2PosEmbed,
)

DEV = 'cuda'
torch.manual_seed(1)
S = 51712        # 2K i2v 单块序列 (主图 25600*2 + txt 512)
HEADS, HDIM, DIM = 48, 128, 6144

pos = Flux2PosEmbed(theta=10000, axes_dim=[32, 32, 32, 32]).to(DEV)
ids = torch.zeros(S, 4, dtype=torch.int64, device=DEV)
idx = torch.arange(S, device=DEV)
ids[:, 1] = idx % 160
ids[:, 2] = (idx // 160) % 160
cos, sin = pos(ids)
print(f'rope cos {tuple(cos.shape)}', flush=True)

attn = Flux2ParallelSelfAttention(
    query_dim=DIM, heads=HEADS, dim_head=HDIM, out_dim=DIM,
    bias=False, out_bias=True, eps=1e-6,
    mlp_ratio=3.0, mlp_mult_factor=2,
    processor=Flux2ParallelSelfAttnProcessor(),
)
attn = attn.to(DEV).eval().bfloat16()
print('parallel attn built', flush=True)

hidden = torch.randn(1, S, DIM, device=DEV, dtype=torch.bfloat16)
with torch.no_grad():
    t0 = time.perf_counter()
    out = attn(hidden_states=hidden, image_rotary_emb=(cos, sin))
    torch.cuda.synchronize()
    print(f'S={S} PARALLEL_ATTN OK ({time.perf_counter()-t0:.2f}s) out={tuple(out.shape)}', flush=True)
print('MINI2_DONE', flush=True)
