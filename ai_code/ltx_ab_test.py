"""LTX 手写引擎 — 128 vs 256 latent A/B 对比诊断脚本
定位 256 出白屏的环节: RoPE / pack_latents / transformer / VAE decode 逐层隔离。

用法:
  python ltx_ab_test.py --stage sanity
     免显卡/免shards 快检: pack/unpack 往返一致性 + RoPE 形状统计 +
     手写 apply_rotary vs 官方 apply_rotary_emb 数值交叉验证 (~秒级)

  python ltx_ab_test.py --stage trajectory --w 256 --h 256 --frames 9 --steps 50 --seed 42
     跑手写引擎去噪, 逐 step 记录 latent 统计, 检查点存 .pt
     关键: 128 和 256 用同一 --mu 跑, 消除 scheduler shift 混淆 (见 print 提示)

  python ltx_ab_test.py --stage compare --a <128.pt> --b <256.pt>
     对比两个最终 latent: 全局/逐通道统计 + 逐 token 能量图 + 异常点定位

  python ltx_ab_test.py --stage decode --lat <256.pt> --w 256 --h 256 --frames 9 [--no_tiling]
     单独跑解码路径, 隔离 "latent 坏" 还是 "decode 坏"
"""
import sys, io, os, time, math, json, argparse
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
import numpy as np

MODEL = r'D:\ai\models\LTX-Video'
OUT = r'D:\ai\output\ab_test'
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda'

from ltx_gen import pack_latents, unpack_latents          # 手写 pack/unpack
from ltx_sched import compute_rope, apply_rotary          # 手写 RoPE 应用
from diffusers.models.transformers.transformer_ltx import apply_rotary_emb as OFFICIAL_APPLY_ROPE

T = lambda s: f'{s:.4f}'


# ══════════════════════ Stage 1: sanity ══════════════════════
def stage_sanity():
    print('═' * 70)
    print('[sanity] Stage1: pack/unpack 往返 + RoPE 形状 + apply_rotary 交叉验证')
    print('═' * 70)

    # ── 1a. pack/unpack 往返一致性 ──
    for name, (w, h, frames) in [('128', (128, 128, 9)), ('256', (256, 256, 9))]:
        lat_F = (frames - 1) // 8 + 1
        lat_H, lat_W = h // 32, w // 32
        x = torch.randn(1, 128, lat_F, lat_H, lat_W, dtype=torch.float32)
        pk = pack_latents(x)
        uk = unpack_latents(pk, frames, h, w)
        err = (uk - x).abs().max().item()
        print(f'[sanity] {name}: latent {lat_F}x{lat_H}x{lat_W}  '
              f'pack->{tuple(pk.shape)} unpack->{tuple(uk.shape)}  往返max_err={err:.2e} '
              f'{"✓ 恒等" if err < 1e-6 else "✗ 数据损坏!"}')

    # ── 1b. RoPE 生成 (两种分辨率) ──
    print('-' * 70)
    rope_info = {}
    for name, (w, h, frames) in [('128', (128, 128, 9)), ('256', (256, 256, 9))]:
        lat_F = (frames - 1) // 8 + 1
        lat_H, lat_W = h // 32, w // 32
        fake = torch.zeros(2, lat_F * lat_H * lat_W, 4096)  # B=2 (CFG)
        rope = compute_rope(fake, lat_F, lat_H, lat_W, 8, 32, 25)
        cos, sin = rope
        ok = (cos.isfinite().all() and sin.isfinite().all()).item()
        rope_info[name] = dict(shape=tuple(cos.shape), S=lat_F * lat_H * lat_W)
        # 每个 token 的 cos 是否唯一 (位置编码是否真的随位置变化)
        uniq = cos.view(cos.shape[0], cos.shape[1], -1)[0].unique(dim=0).shape[0]
        print(f'[sanity] RoPE {name}: cos{tuple(cos.shape)} sin{tuple(sin.shape)} '
              f'finite={ok} 唯一位置数={uniq}/{cos.shape[1]}  '
              f'{"" if uniq == cos.shape[1] else "⚠ 位置编码有重复!"}')
    print(f'[sanity] RoPE 尺寸随分辨率: {rope_info["128"]["S"]} -> {rope_info["256"]["S"]} tokens '
          f'(与 packed latent 的 S 一一对应)')

    # ── 1c. 手写 apply_rotary vs 官方 apply_rotary_emb (数值交叉验证) ──
    print('-' * 70)
    for name, (w, h, frames) in [('128', (128, 128, 9)), ('256', (256, 256, 9))]:
        lat_F = (frames - 1) // 8 + 1
        lat_H, lat_W = h // 32, w // 32
        torch.manual_seed(0)
        q = torch.randn(2, lat_F * lat_H * lat_W, 4096, dtype=torch.bfloat16)
        rope = compute_rope(q, lat_F, lat_H, lat_W, 8, 32, 25)
        a = apply_rotary(q, rope)                    # 手写
        b = OFFICIAL_APPLY_ROPE(q, rope)             # 官方
        d = (a.float() - b.float()).abs().max().item()
        print(f'[sanity] apply_rotary 交叉验证 {name}: 手写vs官方 max_diff={d:.2e} '
              f'{"✓ 一致" if d < 1e-3 else "✗ 不一致!"}')

    # ── 1d. scheduler mu 混淆检查 ──
    print('-' * 70)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler')
    for name, S in [('128', 32), ('256', 128)]:
        mu = calculate_shift(S, sch.config.get('base_image_seq_len', 256),
                             sch.config.get('max_image_seq_len', 4096),
                             sch.config.get('base_shift', 0.5), sch.config.get('max_shift', 1.15))
        print(f'[sanity] seq_len={S:3d} 自然 mu={mu:.4f}   '
              f'<-- {"这两个 mu 不同, 直接对比会引入 timestep 混淆" if name=="256" else ""}')
    print('[sanity] 提示: trajectory 阶段请给 128 和 256 传相同的 --mu, 消除此混淆')

    print('═' * 70)


# ══════════════════════ Stage 2: trajectory ══════════════════════
def stage_trajectory(w, h, frames, steps, seed, mu_override, quick):
    t0 = time.perf_counter()
    lat_F = (frames - 1) // 8 + 1
    lat_H, lat_W = h // 32, w // 32
    seq_len = lat_F * lat_H * lat_W
    tag = f'{w}x{h}_f{frames}'
    if mu_override is not None:
        tag += f'_mu{mu_override:.4f}'
    print(f'[traj] {tag}: latent {lat_F}x{lat_H}x{lat_W} seq={seq_len} steps={steps} seed={seed}')

    # T5 embeds (缓存)
    from ltx_gen import generate_video  # 仅为复用 cache 路径; 不调用
    cache_path = r'D:\ai\_cached_embeds.pt'
    emb_data = torch.load(cache_path, weights_only=True)
    pos = emb_data['pos'].bfloat16(); neg = emb_data['neg'].bfloat16()
    pos_mask = emb_data['pos_mask'].bool(); neg_mask = emb_data['neg_mask'].bool()
    emb_cat = torch.cat([neg, pos], dim=0).to(DEV)
    mask = torch.cat([neg_mask, pos_mask], dim=0).to(DEV)
    print(f'[traj] embeds ready {time.perf_counter() - t0:.0f}s', flush=True)

    # scheduler + timesteps (mu 可覆盖)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler')
    sigmas = np.linspace(1.0, 1 / steps, steps)
    mu_nat = calculate_shift(seq_len, sch.config.get('base_image_seq_len', 256),
                             sch.config.get('max_image_seq_len', 4096),
                             sch.config.get('base_shift', 0.5), sch.config.get('max_shift', 1.15))
    mu = mu_override if mu_override is not None else mu_nat
    timesteps, _ = retrieve_timesteps(sch, steps, 'cpu', sigmas=sigmas, mu=mu)
    print(f'[traj] mu 自然={mu_nat:.4f} 使用={mu:.4f}  timesteps[{len(timesteps)}] '
          f'首末={timesteps[0].item():.2f}/{timesteps[-1].item():.3f}', flush=True)

    # 初始 latent
    torch.manual_seed(seed)
    lat5 = torch.randn(1, 128, lat_F, lat_H, lat_W, dtype=torch.float32)
    latents = pack_latents(lat5).to(DEV)

    from ltx_sched import transformer_forward_sched
    stats = []
    ckpts = {}
    n_ckpt = 5 if not quick else 3
    ckpt_idxs = set(int(i * (steps - 1) / (n_ckpt - 1)) for i in range(n_ckpt))
    with torch.inference_mode():
        for i, t in enumerate(timesteps):
            li = torch.cat([latents, latents], dim=0).bfloat16()
            ts = t.expand(2).to(DEV)
            rope = compute_rope(li, lat_F, lat_H, lat_W, 8, 32, 25)
            np_ = transformer_forward_sched(li, emb_cat, ts, mask, lat_F, lat_H, lat_W,
                                            rope_emb=rope, dtype=torch.bfloat16)
            u, c = np_.chunk(2)
            np_ = u + 3.0 * (c - u)
            latents = sch.step(np_.float(), t, latents.float(), return_dict=False)[0]
            lat = latents.float()
            stats.append(dict(step=i, t=t.item(),
                              mean=lat.mean().item(), std=lat.std().item(),
                              min=lat.min().item(), max=lat.max().item(),
                              absmean=lat.abs().mean().item()))
            if i in ckpt_idxs:
                ckpts[f'step{i:03d}'] = lat.cpu().clone()
            if i % 10 == 0 or i == steps - 1:
                print(f'[traj] step {i + 1}/{steps} t={t.item():.1f} '
                      f'std={lat.std().item():.3f} mean={lat.mean().item():+.3f} '
                      f'[{lat.min().item():.1f},{lat.max().item():.1f}] '
                      f'{time.perf_counter() - t0:.0f}s', flush=True)

    from ltx_sched import cleanup
    cleanup()

    # 存结果
    pk = latents.float().cpu().clone()
    uk = unpack_latents(pk, frames, h, w)
    torch.save(dict(meta=dict(w=w, h=h, frames=frames, steps=steps, seed=seed, mu=mu),
                    packed=pk, unpacked=uk, ckpts=ckpts, stats=stats),
               os.path.join(OUT, f'lat_{tag}.pt'))
    with open(os.path.join(OUT, f'traj_{tag}.json'), 'w') as f:
        json.dump(dict(meta=dict(w=w, h=h, frames=frames, steps=steps, seed=seed, mu=mu),
                       stats=stats), f, indent=1)
    print(f'[traj] saved {OUT}/lat_{tag}.pt + traj_{tag}.json 总耗时 {time.perf_counter() - t0:.0f}s', flush=True)


# ══════════════════════ Stage 3: compare ══════════════════════
def stage_compare(path_a, path_b):
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    print('═' * 70)
    print('[cmp] 全局统计对比')
    print('═' * 70)
    print(f'{"":>14} {"128(mean/std)":>22} {"256(mean/std)":>22}')
    for key in ('std', 'mean', 'absmean'):
        va = a['stats'][-1][key]; vb = b['stats'][-1][key]
        print(f'{key:>12}: {va:>12.4f} {vb:>16.4f}')
    print(f'{"min":>12}: {a["stats"][-1]["min"]:>12.4f} {b["stats"][-1]["min"]:>16.4f}')
    print(f'{"max":>12}: {a["stats"][-1]["max"]:>12.4f} {b["stats"][-1]["max"]:>16.4f}')

    # 逐通道统计 (C=128)
    pa = a['packed']; pb = b['packed']
    print('\n[cmp] 逐通道 std (128 通道):')
    sa = pa.view(-1, pa.shape[-1]).std(0)
    sb = pb.view(-1, pb.shape[-1]).std(0)
    print(f'  通道std: 128 range=[{sa.min():.3f},{sa.max():.3f}] mean={sa.mean():.3f}')
    print(f'  通道std: 256 range=[{sb.min():.3f},{sb.max():.3f}] mean={sb.mean():.3f}')
    # 通道级 std 直方图差异: 用 log 比值
    ratio = (sb + 1e-6) / (sa + 1e-6)
    print(f'  通道std比值(256/128): median={ratio.median():.3f} 异常通道(偏离>2x或<0.5x): '
          f'{(ratio > 2).sum().item()}↑ {(ratio < 0.5).sum().item()}↓')

    # 逐 token 能量图 (spatial 分布)
    print('\n[cmp] 逐 token (f,h,w) 能量图 — 检查空间各位置是否都被正确激活:')
    def energy_map(packed, F, H, W):
        x = packed.view(1, F, H, W, -1)
        return x.std(-1)[0]  # (F,H,W)
    ea = energy_map(pa, a['meta']['frames'] // 8 + 1, a['meta']['h'] // 32, a['meta']['w'] // 32)
    eb = energy_map(pb, b['meta']['frames'] // 8 + 1, b['meta']['h'] // 32, b['meta']['w'] // 32)
    ma = [round(ea[f].mean().item(), 3) for f in range(ea.shape[0])]
    mb = [round(eb[f].mean().item(), 3) for f in range(eb.shape[0])]
    print(f'  128 能量图 (F,H,W)={tuple(ea.shape)}:  std范围=[{ea.min():.3f},{ea.max():.3f}] 各帧均值={ma}')
    print(f'  256 能量图 (F,H,W)={tuple(eb.shape)}:  std范围=[{eb.min():.3f},{eb.max():.3f}] 各帧均值={mb}')

    # 128 vs 256 的逐通道能量分布相似度 (把 256 池化到 128 尺寸)
    eb_pool = torch.nn.functional.adaptive_avg_pool2d(eb, (ea.shape[1], ea.shape[2]))
    for f in range(min(ea.shape[0], eb_pool.shape[0])):
        corr = torch.corrcoef(torch.stack([ea[f].flatten(), eb_pool[f].flatten()]))[0, 1].item()
        print(f'  帧{f} 能量图相关性(256池化后): r={corr:.3f}')

    # 异常点检查
    print('\n[cmp] 异常点检查 (NaN/Inf/极端值):')
    for nm, p in [('128', pa), ('256', pb)]:
        nan = (~torch.isfinite(p)).sum().item()
        extreme = (p.abs() > 5).sum().item()
        print(f'  {nm}: NaN/Inf={nan}  |x|>5 的 token 元素={extreme}')

    # 检查点轨迹对比 (若存在)
    if 'ckpts' in a and 'ckpts' in b and a['ckpts'] and b['ckpts']:
        print('\n[cmp] 检查点轨迹 (同一 mu 下应可比):')
        print(f'{"ckpt":>10} {"128 std":>12} {"256 std":>12} {"128 mean":>12} {"256 mean":>12}')
        for k in sorted(a['ckpts']):
            if k in b['ckpts']:
                va = a['ckpts'][k].float(); vb = b['ckpts'][k].float()
                print(f'{k:>10} {va.std().item():>12.4f} {vb.std().item():>12.4f} '
                      f'{va.mean().item():>12.4f} {vb.mean().item():>12.4f}')
    print('═' * 70)


# ══════════════════════ Stage 4: decode ══════════════════════
def stage_decode(lat_path, w, h, frames, no_tiling):
    data = torch.load(lat_path, weights_only=False)
    lat5 = data['unpacked']
    print(f'[dec] 解码 {lat_path} -> {tuple(lat5.shape)} (原 {data["meta"]})')

    from vae_loader import load_vae_pread
    vae = load_vae_pread(f'{MODEL}/vae_1024')
    m = vae.latents_mean.view(1, -1, 1, 1, 1).to(lat5.dtype)
    s = vae.latents_std.view(1, -1, 1, 1, 1).to(lat5.dtype)
    lat_denorm = lat5 * s / vae.config.scaling_factor + m
    print(f'[dec] denorm: scaling_factor={vae.config.scaling_factor} latents_std mean={s.mean():.4f} '
          f'denorm 后 std={lat_denorm.std():.3f}')

    if not no_tiling:
        vae.enable_tiling()
        print('[dec] tiling=ON (复现手写引擎行为)')
    else:
        print('[dec] tiling=OFF (隔离 tiling 是否致白)')
    vae = vae.to(DEV)
    with torch.inference_mode():
        t_dec = torch.tensor([0.0], device=DEV, dtype=lat_denorm.dtype)
        decoded = vae.decode(lat_denorm.to(DEV), t_dec, return_dict=False)[0].cpu()
    print(f'[dec] decoded {tuple(decoded.shape)}')

    fr = [decoded[0, :, i].permute(1, 2, 0).float() for i in range(decoded.shape[2])]
    fr = [(f + 1) / 2 for f in fr]
    pix = [torch.stack([f.clamp(0, 1) * 255 for f in fr], dim=0)]
    for i, f in enumerate(fr):
        print(f'[dec] 帧{i}: mean={f.mean().item():.3f} std={f.std().item():.3f} '
              f'min={f.min().item():.3f} max={f.max().item():.3f} '
              f'{"⚠ 疑似白屏(均值≈0.5)" if abs(f.mean().item() - 0.5) < 0.1 else "✓"}')

    fn = os.path.join(OUT, f'dec_{os.path.basename(lat_path)[:-3]}_{"notile" if no_tiling else "tile"}.mp4')
    from ltx_gen import save_video_mp4
    from PIL import Image
    imgs = [Image.fromarray((f.clamp(0, 1) * 255).to(torch.uint8).numpy()) for f in fr]
    save_video_mp4(imgs, fn, fps=25)
    print(f'[dec] saved {fn}  {imgs[0].size}')


# ══════════════════════ main ══════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['sanity', 'trajectory', 'compare', 'decode'])
    ap.add_argument('--w', type=int, default=256)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--frames', type=int, default=9)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mu', type=float, default=None, help='覆盖 scheduler shift (128/256 传同值以消除混淆)')
    ap.add_argument('--quick', action='store_true', help='减少检查点数, 快速 A/B')
    ap.add_argument('--a', type=str)
    ap.add_argument('--b', type=str)
    ap.add_argument('--lat', type=str)
    ap.add_argument('--no_tiling', action='store_true')
    args = ap.parse_args()

    if args.stage == 'sanity':
        stage_sanity()
    elif args.stage == 'trajectory':
        stage_trajectory(args.w, args.h, args.frames, args.steps, args.seed, args.mu, args.quick)
    elif args.stage == 'compare':
        stage_compare(args.a, args.b)
    elif args.stage == 'decode':
        stage_decode(args.lat, args.w, args.h, args.frames, args.no_tiling)
