"""LTX-2.5 完整 DiT 采样 (1024 token Gemma4 text_emb) → latent
流程: text_emb(1,1024,4096) → patchify latent → LTX2Scheduler → 48块 DiT → CFG velocity → Euler
输出: latent (1,128,F,H,W) 保存
"""
import sys, io, os, gc, math, time, torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE'] = '1'
import ltx25_sched as s
from ltx_core.components.schedulers import LTX2Scheduler

DEV = 'cuda'
# 参数
F, H, W = 9, 16, 16          # 512x512, 9帧 (latent 16x16)
STEPS = 30
CFG = 3.0
text_emb = torch.load(r'D:\ai\output\ab_test\text_emb_1024.pt', weights_only=True)  # (1,1024,4096)
if isinstance(text_emb, dict):
    text_emb = text_emb.get('text_emb', text_emb.get('video_encoding'))
text_emb = text_emb.float().to(DEV)
B, S_text, D = text_emb.shape
print(f'text_emb: {tuple(text_emb.shape)}', flush=True)
# mask 全 1 (1024 无 pad)
mask = torch.ones(B, S_text, device=DEV)
# 负 prompt: 零向量 (简化, 或空文本编码)
neg_emb = torch.zeros_like(text_emb)

# latent 初始化 + patchify
torch.manual_seed(42)
lat5 = torch.randn(1, 128, F, H, W, dtype=torch.float32, device=DEV)  # (1,128,F,H,W)
latents = lat5.permute(0, 2, 3, 4, 1).reshape(1, F*H*W, 128)  # (1,S,128)
S = F * H * W
print(f'latent: {tuple(lat5.shape)} seq={S}', flush=True)

# positions (正确协议: latent网格 × scale(8,32,32), temporal /fps)
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

positions = build_positions(F, H, W).to(DEV)
pe = s.compute_pe(positions)  # (cos, sin)

# scheduler
sched = LTX2Scheduler()
sigmas = sched.execute(STEPS, lat5, stretch=True, terminal=0.1)
print(f'sigmas: {sigmas.shape} [{sigmas[0]:.3f} ... {sigmas[-1]:.3f}]', flush=True)

@torch.inference_mode()
def run():
    global latents
    t0 = time.time()
    for i in range(STEPS):
        sigma = sigmas[i]
        ts = sigma.expand(2).to(DEV)
        # pos + neg 两路
        inp = torch.cat([latents, latents], dim=0).bfloat16()
        emb_cat = torch.cat([neg_emb, text_emb], dim=0)
        mask_cat = torch.cat([mask, mask], dim=0)
        v = s.transformer_forward_sched(inp, emb_cat, ts, mask_cat, num_frames=F, height=H, width=W, pe=pe)
        vn, vp = v.chunk(2)  # 各 (1,S,128)
        v = vp + CFG * (vp - vn)  # CFG 合成 (velocity 域)
        # Euler 步进 (velocity 参数化): sample_next = sample + v·dt, dt = sigma_next - sigma
        sigma_next = sigmas[i + 1]
        latents = latents + (sigma_next - sigma) * v.float()
        if i % 10 == 0 or i == STEPS - 1:
            print(f'step {i:2d}/{STEPS} sigma={sigma:.3f} latent std={latents.float().std().item():.4f} ({time.time()-t0:.0f}s)', flush=True)
        s.cleanup()
    # 还原 5D
    lat5_out = latents.reshape(1, F, H, W, 128).permute(0, 4, 1, 2, 3).float().cpu()
    os.makedirs(r'D:\ai\output\ab_test', exist_ok=True)
    torch.save(lat5_out, r'D:\ai\output\ab_test\lat_1024_512x512_f9.pt')
    print(f'[完成] latent 保存 {tuple(lat5_out.shape)} 总耗时 {time.time()-t0:.0f}s', flush=True)

run()
