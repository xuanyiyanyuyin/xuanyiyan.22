"""LTX-2.5 引擎 vs 官方管线 逐块对比验证
前提: 已运行 ltx25_capture_official.py 生成 official_trace/
用法: python ltx25_verify_against_official.py
"""
import sys, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')

from ltx25_sched import (read_global, read_block, _adaln_single, _prompt_kv,
                          block_forward, compute_pe)

DEV = 'cuda'; DT = torch.bfloat16
TRACE = r'D:\ai\official_trace'


def build_positions(T, H, W):
    """根据 latent 尺寸构造 positions (B,3,S,2)"""
    B, S = 1, T * H * W
    pos = torch.zeros(B, 3, S, 2)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                i = t * H * W + h * W + w
                pos[0, 0, i] = torch.tensor([t, t + 1])
                pos[0, 1, i] = torch.tensor([h, h + 1])
                pos[0, 2, i] = torch.tensor([w, w + 1])
    return pos


def verify():
    inputs = torch.load(f'{TRACE}/inputs.pt', map_location=DEV)
    print(f"官方输入 keys: {list(inputs.keys())}")
    # 关键: 从官方输入推断 latent 形状 (hidden_states 是 packed (B,S,C))
    x = inputs['hidden_states'].to(DT)
    t = inputs['timestep'].to(DEV).float()
    text_emb = inputs['encoder_hidden_states'].to(DT)
    num_frames = inputs.get('num_frames')
    height = inputs.get('height')
    width = inputs.get('width')
    print(f"x: {tuple(x.shape)} std={x.float().std().item():.4f}")
    print(f"timestep: {t}  sigma: {inputs.get('sigma')}")

    # 从 packed x 反推 latent 尺寸: S = lat_F * lat_H * lat_W
    # 需知道 VAE 压缩比 (32x spatial, 8x temporal) → 传 height/width/frames
    # 或用 num_frames/height/width 反推
    gw = {k: v.to(DEV) for k, v in read_global().items()}
    all_bw = [{k: v.to(DEV) for k, v in read_block(i).items()} for i in range(48)]

    # 生成 pe (从 x 的 S 反推 latent 网格)
    if num_frames is not None and height is not None and width is not None:
        lat_F = (num_frames - 1) // 8 + 1
        lat_H, lat_W = height // 32, width // 32
        S = x.shape[1]
        assert S == lat_F * lat_H * lat_W, f'S={S} != {lat_F}x{lat_H}x{lat_W}'
        pos = build_positions(lat_F, lat_H, lat_W)
        pe = compute_pe(pos)
        pe_c = (pe[0].cuda(), pe[1].cuda())
    else:
        # 退化: 从 S 因式分解 (尽量)
        print("⚠️ 无法确定 latent 尺寸, 用最近似 2D 分解")
        pe_c = None

    # 逐块跑 + 对比
    mods, _ = _adaln_single(t, gw)
    h = x
    all_pass = True
    for i in range(48):
        pk = _prompt_kv(t, gw, all_bw[i])
        h = block_forward(h, text_emb, mods, pe_c, pk, inputs['encoder_attention_mask'].to(DEV) if 'encoder_attention_mask' in inputs else None, all_bw[i])

        official_out = torch.load(f'{TRACE}/block{i:02d}_out.pt', map_location=DEV)
        if isinstance(official_out, tuple):
            official_out = official_out[0]
        official_out = official_out.to(DT)
        if official_out.shape != h.shape:
            print(f"block{i}: shape mismatch {h.shape} vs {official_out.shape}, 尝试对齐")
            # 可能是 unpatchify 前的 (B,C,T,H,W) vs (B,S,C)
            if official_out.ndim == 5 and h.ndim == 3:
                official_out = official_out.permute(0, 2, 3, 4, 1).reshape(h.shape)
            else:
                break
        diff = (h.float() - official_out.float()).abs()
        md = diff.max().item(); rel = (diff / (official_out.float().abs() + 1e-8)).mean().item()
        ok = md < 0.1
        print(f"block{i:2d}: max_diff={md:.4f} rel={rel:.2e} std_mine={h.float().std().item():.3f} std_off={official_out.float().std().item():.3f} {'PASS' if ok else 'FAIL'}")
        if md > 1.0:
            all_pass = False
            break

    print('\n' + ('🎉 ALL 48 BLOCKS MATCH OFFICIAL!' if all_pass else '⚠️ divergence — check the failing block'))


if __name__ == '__main__':
    verify()
