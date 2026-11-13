"""手写引擎 vs 官方 transformer — 单步 forward 对比 (256: S=128)
用法: python ltx_tf_compare.py [--w 256 --frames 9 --timestep 500]
内存策略: 先加载官方transformer跑forward → 释放 → 再跑手写引擎 → 对比
"""
import sys, io, os, time, gc, argparse
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
import numpy as np

DEV = 'cuda'
MODEL = r'D:\ai\models\LTX-Video'

def get_embeds():
    d = torch.load(r'D:\ai\_cached_embeds.pt', weights_only=True)
    pos = d['pos'].bfloat16(); neg = d['neg'].bfloat16()
    pos_mask = d['pos_mask'].bool(); neg_mask = d['neg_mask'].bool()
    return torch.cat([neg, pos], dim=0), torch.cat([neg_mask, pos_mask], dim=0)

def make_input(w, h, frames, ts_val, seed=42):
    lat_F = (frames - 1) // 8 + 1; lat_H = h // 32; lat_W = w // 32
    torch.manual_seed(seed)
    lat = torch.randn(2, lat_F * lat_H * lat_W, 128, dtype=torch.bfloat16)  # B=2 CFG
    emb, mask = get_embeds()
    ts = torch.tensor([ts_val, ts_val])
    return lat, emb, mask, ts, lat_F, lat_H, lat_W

def run_official(lat, emb, mask, ts, lat_F, lat_H, lat_W, scale=(8/25, 32, 32)):
    from diffusers import LTXVideoTransformer3DModel
    t0 = time.perf_counter()
    tf = LTXVideoTransformer3DModel.from_single_file(
        f'{MODEL}/ltxv-13b-0.9.8-dev.safetensors', torch_dtype=torch.bfloat16,
        config=r'D:\ai\ltx_config', low_cpu_mem_usage=True)
    print(f'[cmp] 官方transformer加载 {time.perf_counter()-t0:.0f}s', flush=True)
    tf = tf.to(DEV)
    with torch.inference_mode():
        out = tf(
            hidden_states=lat.to(DEV),
            encoder_hidden_states=emb.to(DEV),
            timestep=ts.to(DEV),
            encoder_attention_mask=mask.to(DEV),
            num_frames=lat_F, height=lat_H, width=lat_W,
            rope_interpolation_scale=scale, return_dict=False,
        )[0].float().cpu()
    print(f'[cmp] 官方forward完成 {tuple(out.shape)} {time.perf_counter()-t0:.0f}s', flush=True)
    del tf; gc.collect(); torch.cuda.empty_cache()
    return out

def run_hand(lat, emb, mask, ts, lat_F, lat_H, lat_W):
    from ltx_sched import transformer_forward_sched, compute_rope
    t0 = time.perf_counter()
    rope = compute_rope(lat, lat_F, lat_H, lat_W, 8, 32, 25)
    out = transformer_forward_sched(lat, emb, ts, mask, lat_F, lat_H, lat_W,
                                    rope_emb=rope, dtype=torch.bfloat16)
    print(f'[cmp] 手写forward完成 {tuple(out.shape)} {time.perf_counter()-t0:.0f}s', flush=True)
    from ltx_sched import cleanup
    cleanup()
    return out.cpu()

def main(w, h, frames, ts_val):
    lat, emb, mask, ts, lat_F, lat_H, lat_W = make_input(w, h, frames, ts_val)
    print(f'[cmp] 输入: latent(2,{lat.shape[1]},128) emb{tuple(emb.shape)} ts={ts_val} F={lat_F} H={lat_H} W={lat_W}', flush=True)

    off = run_official(lat, emb, mask, ts, lat_F, lat_H, lat_W)
    hand = run_hand(lat, emb, mask, ts, lat_F, lat_H, lat_W)

    diff = (off - hand).abs()
    rel = diff / (off.abs() + 1e-6)
    print('═' * 60)
    print(f'[cmp] 对比 (S={lat.shape[1]}):')
    print(f'  官方: mean={off.mean().item():+.4f} std={off.std().item():.4f}')
    print(f'  手写: mean={hand.mean().item():+.4f} std={hand.std().item():.4f}')
    print(f'  max_abs_diff = {diff.max().item():.6f}')
    print(f'  mean_abs_diff = {diff.mean().item():.6f}')
    print(f'  相对(max) = {rel.max().item():.6f}  相对(mean) = {rel.mean().item():.6f}')
    ok = diff.max().item() < 0.05
    print(f'  → {"✓ 一致" if ok else "✗ 不一致! 手写forward有bug"}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, default=256)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--frames', type=int, default=9)
    ap.add_argument('--ts', type=float, default=500.0)
    a = ap.parse_args()
    main(a.w, a.h, a.frames, a.ts)
