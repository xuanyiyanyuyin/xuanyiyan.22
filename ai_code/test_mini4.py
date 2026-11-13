# -*- coding: utf-8 -*-
"""严格复刻 Flux2ParallelSelfAttnProcessor 每行, 分步sync, 找非法访问第一现场."""
import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2ParallelSelfAttention, Flux2ParallelSelfAttnProcessor, Flux2PosEmbed,
    apply_rotary_emb, dispatch_attention_fn,
)

DEV = 'cuda'
torch.manual_seed(1)
import sys
S = int(sys.argv[1]) if len(sys.argv) > 1 else 51712
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
        out = fn()
        torch.cuda.synchronize()
        print(f'[OK] {name}', flush=True)
        return out
    except Exception as e:
        print(f'[FAIL] {name}: {type(e).__name__}: {str(e)[:80]}', flush=True)
        raise SystemExit

with torch.no_grad():
    h = step('proj', lambda: attn.to_qkv_mlp_proj(hidden))
    qkv_dim = 3 * attn.inner_dim
    mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
    local_qkv = h.shape[-1] * qkv_dim // (qkv_dim + mlp_dim)
    qkv, mlp_hidden = step('split', lambda: torch.split(h, [local_qkv, h.shape[-1] - local_qkv], dim=-1))
    q, k, v = step('chunk', lambda: qkv.chunk(3, dim=-1))
    q = step('unflatten_q', lambda: q.unflatten(-1, (-1, attn.head_dim)))
    k = step('unflatten_k', lambda: k.unflatten(-1, (-1, attn.head_dim)))
    v = step('unflatten_v', lambda: v.unflatten(-1, (-1, attn.head_dim)))
    q = step('norm_q', lambda: attn.norm_q(q))
    k = step('norm_k', lambda: attn.norm_k(k))
    q = step('rope_q', lambda: apply_rotary_emb(q, (cos, sin), sequence_dim=1))
    k = step('rope_k', lambda: apply_rotary_emb(k, (cos, sin), sequence_dim=1))
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    print('ACTIVE_BACKEND =', repr(_AttentionBackendRegistry.get_active_backend()[0]), flush=True)
    print('q contig:', q.is_contiguous(), 'k contig:', k.is_contiguous(), 'v contig:', v.is_contiguous(), flush=True)
    print('q shape:', tuple(q.shape), 'k shape:', tuple(k.shape), 'v shape:', tuple(v.shape), flush=True)
    v = v.contiguous()
    print('v after contiguous:', v.is_contiguous(), flush=True)
    out = step('dispatch(native-explicit, v-contig)', lambda: dispatch_attention_fn(q, k, v, backend='native'))
print('MINI4_DONE', flush=True)
