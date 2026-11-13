"""LTX-2.5 内存预算计算器 — 规划 offload 策略
用法: python ltx25_memory_budget.py --vram 12 --ram 32 --resolution 512x512x9
"""
import argparse, math

def gb(params_b, dtype_bytes=2):
    return params_b * dtype_bytes / 1e9

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vram', type=float, default=12.0)
    p.add_argument('--ram', type=float, default=32.0)
    p.add_argument('--resolution', default='512x512x9')
    args = p.parse_args()

    # LTX-2.5 实际组件 (bf16)
    dit_video_params = 13.1e9    # video-only DiT (删音频后)
    gemma_params = 12e9          # Gemma4-12B 文本编码器
    vae_params = 0.5e9           # DiffVAE 解码路径 ~500M
    connector_params = 1e9       # video_embeddings_connector (含在DiT?)

    # 激活内存 (bf16)
    W, H, F = map(int, args.resolution.split('x'))
    patch_t, patch_h, patch_w = 8, 32, 32
    T = math.ceil(F / patch_t)
    S_spatial = (H // patch_h) * (W // patch_w)
    S = T * S_spatial
    D = 4096
    attn_act = S * S * 64 * 2 / 1e9
    ffn_act = S * D * 4 * 2 / 1e9
    resid_act = S * D * 2 / 1e9
    act_peak = attn_act + ffn_act + resid_act

    print('='*60)
    print(f'LTX-2.5 Memory Budget ({args.resolution})')
    print('='*60)
    print(f'Target: {args.resolution} → S={S} tokens')
    print(f'Activation peak (single step): ~{act_peak:.2f} GB')

    print(f'\nComponents (bf16):')
    comps = {
        'DiT video-only (13.1B)': gb(dit_video_params),
        'Gemma4-12B TE': gb(gemma_params),
        'DiffVAE (~500M)': gb(vae_params),
    }
    total = sum(comps.values())
    for n, s in comps.items():
        bar = '#' * int(s / total * 30)
        print(f'  {n:22s}: {s:6.2f} GB  {bar}')
    print(f'  {"TOTAL":22s}: {total:6.2f} GB')

    print(f'\nAvailable: VRAM {args.vram:.0f} + RAM {args.ram:.0f} = {args.vram+args.ram:.0f} GB')
    deficit = total + act_peak - (args.vram + args.ram)
    print(f'Deficit (all-in-memory): {max(0, deficit):.2f} GB')

    print(f'\n=== Recommended Offload Strategy (你的三级加载) ===')
    print(f'  [1] Gemma4 TE: 加载→encode一次→释放 (RAM峰值+24GB, 短时)')
    print(f'  [2] DiT: 磁盘→RAM→GPU 逐块 (每块~{gb(dit_video_params/48):.2f}GB上GPU)')
    print(f'  [3] DiffVAE: 解码时才加载 (~1GB GPU 或 CPU)')
    per_block = gb(dit_video_params) / 48
    gpu_dit = max(0, args.vram - 2)  # 留2GB给激活/VAE
    print(f'  每块 {per_block:.2f}GB, GPU可容纳 ~{int(gpu_dit/per_block)} 块常驻')

    print(f'\n=== 量化选项 (如需) ===')
    for qn, qb, qq in [('INT8/FP8', 1, 'Good'), ('INT4', 0.5, 'Acceptable')]:
        dq = gb(dit_video_params, qb); gq = gb(gemma_params, qb)
        tq = dq + gq + gb(vae_params)
        fits = 'OK' if tq + act_peak <= args.vram + args.ram else 'NO'
        print(f'  {fits} {qn:10s}: DiT={dq:.1f} Gemma4={gq:.1f} Total={tq:.1f}GB ({qq})')

    print(f'\n=== 冷启动清单 ===')
    print(f'  1. 跑本脚本确认目标分辨率预算')
    print(f'  2. Gemma4 encode (一次, 缓存text_emb)')
    print(f'  3. 用三级加载引擎跑去噪循环')
    print(f'  4. DiffVAE 解码')
    print('='*60)

if __name__ == '__main__':
    main()
