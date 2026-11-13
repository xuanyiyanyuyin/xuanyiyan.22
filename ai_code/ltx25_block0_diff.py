"""LTX-2.5 Block0 子模块级数值对拍
逐层对比 我的实现 vs 官方 BasicAVTransformerBlock
用法: python ltx25_block0_diff.py
"""
import sys, torch
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
sys.path.insert(0, r'D:\ai')

from ltx25_sched import (read_block, read_global, _adaln_single, ada_zero,
                          attention_forward, ff_forward, rms_norm)
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer import TransformerConfig

DEV = 'cuda'; DTYPE = torch.bfloat16
torch.manual_seed(42)

def compare(name, o, m):
    o = o.float(); m = m.float()
    d = (o - m).abs()
    rel = d / (o.abs() + 1e-8)
    ok = 'PASS' if d.max().item() < 1e-3 else 'FAIL'
    print(f'[{ok}] {name}: '
          f'std_o={o.std().item():.4f} std_m={m.std().item():.4f} '
          f'max_diff={d.max().item():.2e} mean={d.mean().item():.2e} '
          f'max_rel={rel.max().item():.2e}')

# ═══ 加载权重 ═══
gw = {k: v.to(DEV) for k, v in read_global().items()}
bw = {k: v.to(DEV) for k, v in read_block(0).items()}

# ═══ 实例化官方 block ═══
video_cfg = TransformerConfig(dim=4096, heads=32, d_head=128, context_dim=4096,
                              apply_gated_attention=True, cross_attention_adaln=True, ff_bias=False)
block = BasicAVTransformerBlock(video=video_cfg, audio=None).to(DEV, dtype=DTYPE).eval()
# 加载权重 (strip 前缀)
for sub, pref in [('attn1', 'attn1'), ('attn2', 'attn2'), ('ff', 'ff')]:
    sub_sd = {k[len(pref) + 1:]: v for k, v in bw.items() if k.startswith(pref + '.')}
    getattr(block, sub).load_state_dict(sub_sd, strict=True)
block.scale_shift_table.data.copy_(bw['scale_shift_table'])
if hasattr(block, 'prompt_scale_shift_table'):
    block.prompt_scale_shift_table.data.copy_(bw['prompt_scale_shift_table'])
block = block.to(DEV, dtype=DTYPE).eval()  # 确保所有参数 bf16
print('官方 block0 权重加载 OK')

# ═══ 共享输入 ═══
B, T, D = 1, 32, 4096
x = torch.randn(B, T, D, device=DEV, dtype=DTYPE) * 0.1
text_emb = torch.randn(B, 16, D, device=DEV, dtype=DTYPE) * 0.1
timestep = torch.tensor([0.5], device=DEV)
mods, _ = _adaln_single(timestep, gw)
bm = mods + block.scale_shift_table.unsqueeze(0)  # (B,9,D)
print(f'输入: x={tuple(x.shape)} mods={tuple(mods.shape)} block_mods={tuple(bm.shape)}')

w1 = {k[6:]: v for k, v in bw.items() if k.startswith('attn1.')}
w2 = {k[6:]: v for k, v in bw.items() if k.startswith('attn2.')}
ffw = {k[3:]: v for k, v in bw.items() if k.startswith('ff.')}

# ═══ 1. ada_zero (self-attn) ═══
sa_shift, sa_scale, sa_gate = bm[:, 0], bm[:, 1], bm[:, 2]
my_ada = ada_zero(x, sa_scale, sa_shift)
off_ada = block.ada_zero_function(x, 1e-6, sa_scale, sa_shift)
compare('ada_zero(sa)', off_ada, my_ada)

# ═══ 2. attn1 输出 (含 gate, 无 RoPE) ═══
x_norm = ada_zero(x, sa_scale, sa_shift)
my_attn = attention_forward(x_norm, None, None, w1, is_self=True)
off_attn = block.attn1(x_norm, pe=None)
compare('attn1(no rope)', off_attn, my_attn)

# ═══ 3. post_sa (残差+gate) ═══
my_post = x + my_attn * sa_gate
off_post, off_normed = block.post_sa_function(x, off_attn, None, 1e-6, sa_gate)
compare('post_sa 残差', off_post, my_post)
compare('post_sa normed', off_normed, rms_norm(my_post, None))

# ═══ 4. 交叉注意力 (无 RoPE, 无 mask) ═══
x_normed = rms_norm(x + my_attn * sa_gate, None)
q_shift, q_scale, q_gate = bm[:, 6], bm[:, 7], bm[:, 8]
attn_in = x_normed * (1 + q_scale) + q_shift
# 官方: apply_cross_attention_adaln
from ltx_core.model.transformer.transformer import apply_cross_attention_adaln
kv_mod = block.prompt_scale_shift_table[None, None].to(x.dtype)
shift_kv, scale_kv = kv_mod.unbind(dim=2)
text_mod = text_emb * (1 + scale_kv) + shift_kv
off_ca = apply_cross_attention_adaln(x_normed, text_emb, block.attn2,
                                     q_shift, q_scale, q_gate,
                                     block.prompt_scale_shift_table, None, None)
my_ca = attention_forward(attn_in, text_mod, None, w2, is_self=False) * q_gate
compare('cross_attn', off_ca, my_ca)

# ═══ 5. FFN ═══
ff_shift, ff_scale, ff_gate = bm[:, 3], bm[:, 4], bm[:, 5]
x_full = x + my_attn * sa_gate + my_ca
my_ff_in = ada_zero(x_full, ff_scale, ff_shift)
my_ff = ff_forward(my_ff_in, ffw)
off_ff_in = block.ada_zero_function(x_full, 1e-6, ff_scale, ff_shift)
off_ff = block.ff(off_ff_in)
compare('FFN 输入(ada_zero)', off_ff_in, my_ff_in)
compare('FFN 输出', off_ff, my_ff)

# ═══ 6. 完整块输出 ═══
my_block_out = x_full + my_ff * ff_gate
off_block_out = x_full + off_ff * ff_gate
compare('块最终输出', off_block_out, my_block_out)
print('\n✅ 对拍完成 — 若 ada_zero/attn1 对不上，后续全是连锁错误，先修前面的')
