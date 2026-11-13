"""官方 diffusers LTX 管线 — 256/512 latent 对照 (与手写引擎 ab_test 对比)
用法: python ltx_official_ab.py --w 256 --h 256 --frames 9 --steps 50 --seed 42
输出: 保存 packed latent + 解码报告
"""
import sys, io, os, time, torch, argparse
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np

MODEL = r'D:\ai\models\LTX-Video'
OUT = r'D:\ai\output\ab_test'

def get_pipe():
    """跳过 T5 (用缓存 embeds), 省 ~10G 内存 """
    from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from diffusers import FlowMatchEulerDiscreteScheduler
    t0 = time.perf_counter()
    tf = LTXVideoTransformer3DModel.from_single_file(
        f'{MODEL}/ltxv-13b-0.9.8-dev.safetensors', torch_dtype=torch.bfloat16,
        config=r'D:\ai\ltx_config', low_cpu_mem_usage=True)
    print(f'[off] transformer loaded {time.perf_counter()-t0:.0f}s', flush=True)
    import gc; gc.collect()
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler')
    print(f'[off] components loaded {time.perf_counter()-t0:.0f}s', flush=True)
    # 只装 transformer (output_type='latent' 不需 VAE), 省 2.5G 内存
    pipe = LTXPipeline(transformer=tf, scheduler=sch, vae=None, text_encoder=None, tokenizer=None)
    try:
        pipe.enable_sequential_cpu_offload(device='cuda')
    except Exception as e:
        print(f'[off] offload跳过({e}); 用手动cpu offload', flush=True)
        tf = tf.to('cpu')
    print(f'[off] offload ready {time.perf_counter()-t0:.0f}s', flush=True)
    return pipe

def main(w, h, frames, steps, seed):
    t0 = time.perf_counter()
    pipe = get_pipe()
    from ltx_gen import unpack_latents  # 复用已验证的 unpack
    # 用缓存 embeds 跳过 T5
    d = torch.load(r'D:\ai\_cached_embeds.pt', weights_only=True)
    pos = d['pos'].bfloat16(); neg = d['neg'].bfloat16()
    pos_mask = d['pos_mask'].bool(); neg_mask = d['neg_mask'].bool()
    gen = torch.Generator('cpu').manual_seed(seed)
    res = pipe(prompt=None, prompt_embeds=pos, prompt_attention_mask=pos_mask,
               negative_prompt_embeds=neg, negative_prompt_attention_mask=neg_mask,
               num_inference_steps=steps, num_frames=frames,
               height=h, width=w, guidance_scale=3.0, output_type='latent', generator=gen)
    lat = res.frames
    if isinstance(lat, (list, tuple)): lat = lat[0]
    print(f'[off] latent packed {tuple(lat.shape)} {time.perf_counter()-t0:.0f}s', flush=True)
    print(f'[off] 轨迹? 官方只给最终latent. final std={lat.float().std().item():.3f} mean={lat.float().mean().item():+.3f}', flush=True)

    tag = f'off_{w}x{h}_f{frames}'
    torch.save(dict(meta=dict(w=w, h=h, frames=frames, steps=steps, seed=seed), packed=lat.cpu()),
               os.path.join(OUT, f'lat_{tag}.pt'))
    print(f'[off] saved {OUT}/lat_{tag}.pt  packed{tuple(lat.shape)}  总耗时 {time.perf_counter()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, default=256)
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--frames', type=int, default=9)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    main(a.w, a.h, a.frames, a.steps, a.seed)
