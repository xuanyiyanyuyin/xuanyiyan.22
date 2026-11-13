"""LTX-2.5 图生视频 (i2v) — 图片首帧 + 1024 token Gemma4 text_emb → 视频
机制(照官方): VAE encoder 编码图片 → image_latent 替换第一帧 latent → 每步 Euler 后第一帧重置(denoise_mask=0)
用法: python ltx25_e2e_i2v.py [image.png] [prompt]
"""
import sys, io, os, gc, math, time, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-pipelines\src')
os.environ['HF_HUB_OFFLINE'] = '1'
import ltx25_sched as s
from ltx_core.components.schedulers import LTX2Scheduler

DEV = 'cuda'
VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'
# 参数: 2560×1408 (2K 原生, 不超分) → latent 80×44，对齐文生视频 50 步
# 命令行可调: python ltx25_e2e_i2v.py [image] [prompt] [steps] [frames] [width] [height]
W, H = 2560, 1408
F = 9          # 帧数 (latent)
STEPS = 50     # 对齐文生视频 (gen_video.py 默认 50 步)
CFG = 3.0

def load_encoder():
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER
    builder = SingleGPUModelBuilder(
        model_class_configurator=VideoEncoderConfigurator,
        model_path=VAE_FILE,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    )
    return builder.build(dtype=torch.bfloat16)

def encode_image(encoder, img_path):
    """图片 → VAE encoder → image_latent (B,128,1,HL,WL)"""
    from PIL import Image
    import numpy as np
    img = Image.open(img_path).convert('RGB').resize((W, H), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1,1]
    t = torch.tensor(arr, device=DEV, dtype=torch.bfloat16).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1,3,1,H,W)
    with torch.inference_mode():
        lat = encoder(t)  # (1,128,1,HL,WL)
    print(f'image_latent: {tuple(lat.shape)}', flush=True)
    return lat

def build_positions(T, Hh, Ww):
    B, SS = 1, T * Hh * Ww
    pos = torch.zeros(B, 3, SS, 2)
    for t in range(T):
        for h in range(Hh):
            for w in range(Ww):
                i = t * Hh * Ww + h * Ww + w
                pos[0, 0, i] = torch.tensor([t * 8, (t + 1) * 8], dtype=torch.float32) / 24.0
                pos[0, 1, i] = torch.tensor([h * 32, (h + 1) * 32], dtype=torch.float32)
                pos[0, 2, i] = torch.tensor([w * 32, (w + 1) * 32], dtype=torch.float32)
    return pos

def main():
    global F, STEPS, W, H
    img_path = sys.argv[1] if len(sys.argv) > 1 else r'D:\ai\output\images\flux_verify_2k.png'
    prompt = sys.argv[2] if len(sys.argv) > 2 else 'a cute orange cat sitting on a wooden table'
    if len(sys.argv) > 3: STEPS = int(sys.argv[3])   # 步数 (默认50, 可降到15加速)
    if len(sys.argv) > 4: F = int(sys.argv[4])       # 帧数 (默认9, 可降到5加速)
    if len(sys.argv) > 5: W = int(sys.argv[5])       # 宽 (默认2560 2K)
    if len(sys.argv) > 6: H = int(sys.argv[6])       # 高 (默认1408)
    print(f'[i2v] 参数: {W}x{H}, F={F}, STEPS={STEPS}, CFG={CFG}', flush=True)
    t0 = time.time()
    out = None  # latent 文件路径, 由 run() 赋值后供 VAE decode 使用

    # 1. 图片编码
    encoder = load_encoder()
    image_latent = encode_image(encoder, img_path)  # (1,128,1,HL,WL)
    HL, WL = image_latent.shape[3], image_latent.shape[4]
    print(f'latent 网格: F={F} H={HL} W={WL} (像素 {W}x{H})', flush=True)

    # 2. text_emb (1024 token Gemma4)
    emb_file = r'D:\ai\output\ab_test\text_emb_1024.pt'
    if os.path.exists(emb_file):
        text_emb = torch.load(emb_file, weights_only=True)
        if isinstance(text_emb, dict):
            text_emb = text_emb.get('text_emb', text_emb.get('video_encoding'))
        text_emb = text_emb.float().to(DEV)
    else:
        text_emb = torch.randn(1, 1024, 4096, device=DEV) * 0.1
    neg_emb = torch.zeros_like(text_emb)
    mask = torch.ones(1, text_emb.shape[1], device=DEV)

    # 3. 初始 latent: 第一帧 = 图片, 其他帧 = 噪声
    torch.manual_seed(42)
    lat5 = torch.randn(1, 128, F, HL, WL, dtype=torch.float32, device=DEV)
    lat5[:, :, 0:1, :, :] = image_latent.float()  # 第一帧替换
    latents = lat5.permute(0, 2, 3, 4, 1).reshape(1, F*HL*WL, 128)  # (1,S,128)
    S = F * HL * WL
    first_frame_tokens = HL * WL  # 第一帧 token 数
    img_flat = image_latent.permute(0, 2, 3, 4, 1).reshape(1, HL*WL, 128)  # (1,HL*WL,128)

    positions = build_positions(F, HL, WL).to(DEV)
    pe = s.compute_pe(positions)

    sched = LTX2Scheduler()
    sigmas = sched.execute(STEPS, lat5, stretch=True, terminal=0.1)
    print(f'sigmas {sigmas.shape} tokens={S}', flush=True)

    @torch.inference_mode()
    def run():
        nonlocal latents, out
        for i in range(STEPS):
            sigma = sigmas[i]
            ts = sigma.expand(2).to(DEV)
            inp = torch.cat([latents, latents], dim=0).bfloat16()
            emb_cat = torch.cat([neg_emb, text_emb], dim=0)
            mask_cat = torch.cat([mask, mask], dim=0)
            v = s.transformer_forward_sched(inp, emb_cat, ts, mask_cat, num_frames=F, height=HL, width=WL, pe=pe)
            vn, vp = v.chunk(2)
            v = vp + CFG * (vp - vn)
            sigma_next = sigmas[i + 1]
            latents = latents + (sigma_next - sigma) * v.float()
            # i2v: 第一帧重置为图片 (denoise_mask=0 语义)
            latents[:, :first_frame_tokens, :] = img_flat
            if i % 5 == 0 or i == STEPS - 1:
                print(f'step {i:2d}/{STEPS} sigma={sigma:.3f} std={latents.float().std().item():.4f} ({time.time()-t0:.0f}s)', flush=True)
            s.cleanup()
        lat5_out = latents.reshape(1, F, HL, WL, 128).permute(0, 4, 1, 2, 3).float().cpu()
        os.makedirs(r'D:\ai\output\ab_test', exist_ok=True)
        out = f'D:\\ai\\output\\ab_test\\lat_i2v_{W}x{H}_f{F}.pt'  # 动态文件名, 避免写死2K
        torch.save(lat5_out, out)
        print(f'[完成] latent 保存 {tuple(lat5_out.shape)} 总耗时 {time.time()-t0:.0f}s', flush=True)

    run()  # 生成 latent 并保存到 out

    # 自动 VAE decode → mp4 (自动化: 一步出视频)
    import ltx25_vae_decode_1024 as dec
    dec.main(out)
    print(f'[完成] 图生视频 mp4: D:\\ai\\output\\ab_test\\lat_i2v_{W}x{H}_f{F}.mp4', flush=True)

if __name__ == '__main__':
    main()
