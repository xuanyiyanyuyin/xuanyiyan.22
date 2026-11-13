# -*- coding: utf-8 -*-
"""最小复现: S=51712 直接用真实单块 Flux2Attention + Flux2PosEmbed(rope) 裸跑.
不加载权重, 随机输入. 复制引擎 _fwd 的单块 attn 形态."""
import torch, time
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention, Flux2PosEmbed, Flux2AttnProcessor,
)

DEV = 'cuda'
torch.manual_seed(1)
S = 51712        # 2K i2v: img 25600*2 + txt 512
HEADS, HDIM, DIM = 48, 128, 6144

# rope: 与引擎/官方一致 (Flux2PosEmbed, theta=10000, axes 4x32)
pos = Flux2PosEmbed(theta=10000, axes_dim=[32, 32, 32, 32]).to(DEV)
ids = torch.zeros(S, 4, dtype=torch.int64, device=DEV)
idx = torch.arange(S, device=DEV)
ids[:, 1] = idx % 160
ids[:, 2] = (idx // 160) % 160
ids[:, 3] = 0
cos, sin = pos(ids)
print(f'rope cos {tuple(cos.shape)} sin {tuple(sin.shape)}', flush=True)

# 单块 attention (真实构造参数)
attn = Flux2Attention(query_dim=DIM, added_kv_proj_dim=DIM, dim_head=HDIM, heads=HEADS,
                      out_dim=DIM, bias=False, added_proj_bias=False, out_bias=False,
                      eps=1e-6, processor=Flux2AttnProcessor())
attn = attn.to(DEV).eval().bfloat16()
print('attn built', flush=True)

hidden = torch.randn(1, S, DIM, device=DEV, dtype=torch.bfloat16)
with torch.no_grad():
    t0 = time.perf_counter()
    out = attn(hidden_states=hidden, image_rotary_emb=(cos, sin))
    torch.cuda.synchronize()
    print(f'S={S} ATT+ROPE OK ({time.perf_counter()-t0:.2f}s) out={tuple(out.shape)}', flush=True)
print('MINI_DONE', flush=True)
