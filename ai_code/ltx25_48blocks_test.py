"""LTX-2.5 48块串联 forward 测试 — std 曲线健康检查
用法: python ltx25_48blocks_test.py
"""
import sys, time, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')

from ltx25_sched import (read_global, read_block, _adaln_single, _prompt_kv,
                          block_forward, compute_pe)

DEV = 'cuda'; DTYPE = torch.bfloat16
torch.manual_seed(42)

print('加载权重...', flush=True)
gw = {k: v.to(DEV) for k, v in read_global().items()}
print(f'global: {len(gw)} tensors', flush=True)
# 构造输入: 小 latent 2x4x4 = 32 tokens (和 block0 对拍一致)
T, H, W = 2, 4, 4
B = 1; S = T * H * W
x = torch.randn(B, S, 4096, device=DEV, dtype=DTYPE) * 0.1
text_emb = torch.randn(B, 16, 4096, device=DEV, dtype=DTYPE) * 0.1
timestep = torch.tensor([0.5], device=DEV)
mask = torch.ones(B, 16, dtype=torch.bool, device=DEV)

# 生成 pe
pos = torch.zeros(B, 3, S, 2)
for t in range(T):
    for h in range(H):
        for w in range(W):
            i = t * H * W + h * W + w
            pos[0, 0, i] = torch.tensor([t, t + 1])
            pos[0, 1, i] = torch.tensor([h, h + 1])
            pos[0, 2, i] = torch.tensor([w, w + 1])
pe = compute_pe(pos)
pe_c = (pe[0].cuda(), pe[1].cuda())
print(f'pe: {pe[0].shape}', flush=True)

mods, _ = _adaln_single(timestep, gw)
print(f'mods: {mods.shape}', flush=True)

t0 = time.time()
std_curve = []
with torch.no_grad():
    h = x
    for i in range(48):
        bw = {k: v.to(DEV) for k, v in read_block(i).items()}
        pk = _prompt_kv(text_emb, gw, bw)
        h = block_forward(h, text_emb, mods, pe_c, pk, mask, bw)
        std_curve.append(h.float().std().item())
        del bw
        if i % 8 == 7:
            print(f'  block {i:2d}: std={std_curve[-1]:.4f}', flush=True)
elapsed = time.time() - t0

print(f'=' * 50)
print(f'48块 forward 完成: {elapsed:.2f}s')
print(f'输出: shape={tuple(h.shape)} std={h.float().std().item():.4f}')
print(f'std 曲线 (每8块):')
for i in range(0, 48, 8):
    print(f'  block {i:2d}: std={std_curve[i]:.4f}')
print(f'  block 47: std={std_curve[47]:.4f}')
# 健康检查
if any(s > 1e4 for s in std_curve):
    print('WARNING: std 爆炸!')
elif any(torch.isnan(torch.tensor(s)) for s in std_curve):
    print('WARNING: NaN!')
else:
    print('std 曲线有限 (健康)')
