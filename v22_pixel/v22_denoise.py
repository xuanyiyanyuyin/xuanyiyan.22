# -*- coding: utf-8 -*-
"""V22 噪点→原像素 轻量去噪器（先不带文字分支，轻便优先）。
图像知识依据：DnCNN残差学习(预测噪声残差) + FFDNet噪声水平条件 + Charbonnier L1+SSIM(去噪不用CE押色)
            + 真实噪声模型(高斯+泊松+脉冲)。靶=原图原像素，单像素对单像素。
单组、梯度检查点防OOM。"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs
DEV = M.DEV
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"


def add_noise(clean, rng, sigma):
    """真实噪声模型：高斯+泊松+脉冲（clean∈[0,1] tensor）"""
    c = clean.cpu().numpy()
    g = rng.normal(0, sigma, c.shape).astype(np.float32)                 # 高斯
    scale = max(sigma * 2, 0.01)
    p = (rng.poisson(np.clip(c, 0, 1) * scale * 255) / (scale * 255) - c).astype(np.float32) * 0.5  # 泊松
    noisy = c + g + p
    mask = rng.random(c.shape[:2]) < 0.02                                # 脉冲2%
    salt = rng.random(c.shape[:2]) < 0.5
    noisy[mask] = np.where(salt[mask], 1.0, 0.0)[..., None] if noisy.ndim == 3 else np.where(salt[mask], 1.0, 0.0)
    return torch.tensor(np.clip(noisy, 0, 1), dtype=torch.float32)


class ResBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.b(x)


class DenoiseNet(nn.Module):
    """残差学习去噪：输入 noisy+噪声图 → 预测噪声残差 → clean=noisy-residual。"""
    def __init__(self, dim=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(4, dim, 3, padding=1)                      # 3通道noisy + 1噪声水平图(FFDNet)
        blocks = []
        for i in range(n_blocks):
            d = 1 if i < n_blocks // 2 else 2                            # 后半空洞扩感受野(非局部近似)
            blocks.append(ResBlock(dim, d))
        self.blocks = nn.ModuleList(blocks)
        self.tail = nn.Conv2d(dim, 3, 3, padding=1)                      # 预测噪声残差
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)  # 零初始化→起点恒等

    def forward(self, noisy, sigma_map):
        h = self.head(torch.cat([noisy, sigma_map], 1))
        for b in self.blocks:
            h = checkpoint(b, h, use_reentrant=False)
        residual = self.tail(h)
        return (noisy - residual).clamp(0, 1)                            # 残差学习 clean=noisy-noise


def charbonnier(pred, tgt, eps=1e-3):
    return torch.sqrt((pred - tgt) ** 2 + eps ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "pixel_denoise_out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    torch.cuda.set_per_process_memory_fraction(1.0)
    src = V.load_src(SRC); src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)
    model = DenoiseNet(dim=args.dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    print(f"[denoise] 参数 {sum(p.numel() for p in model.parameters()):,} | 残差学习+噪声条件+Charbonnier+SSIM", flush=True)
    sigmas = [0.05, 0.1, 0.15, 0.2]

    def exam(tag):
        model.eval(); res = []
        with torch.no_grad():
            for s in sigmas:
                noisy = add_noise(src_t[0], np.random.default_rng(43), s).unsqueeze(0).to(DEV)
                smap = torch.full((1, 1, 512, 512), s, device=DEV)
                pred = model(noisy, smap)
                p_np = pred[0].permute(1, 2, 0).cpu().numpy(); t_np = src
                mse = float(((p_np - t_np) ** 2).mean())
                psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
                ss = cs.ssim(p_np, t_np)
                de = float(cs.delta_e_2000(cs.rgb_to_lab(p_np), cs.rgb_to_lab(t_np)).mean())
                res.append((s, psnr, ss, de, pred, noisy))
        model.train()
        print(f"  [denoise {tag}] " + " | ".join(f"σ{s}:PSNR={p:.2f}/SSIM={ss:.3f}/ΔE={de:.2f}" for s, p, ss, de, *_ in res), flush=True)
        if tag.startswith("E"):
            for s, p, ss, de, pred, noisy in res:
                trip = torch.cat([noisy[0].cpu(), pred[0].cpu(), src_t[0].cpu()], dim=2).permute(1, 2, 0).numpy()
                big = (cs.floyd_steinberg(np.clip(trip, 0, 1)) * 255).round().astype(np.uint8)
                cs.save_png_with_icc(os.path.join(args.out, f"{tag}_s{s}.png"), big)
        return res

    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        s = sigmas[(ep - 1) % len(sigmas)]
        noisy = add_noise(src_t[0], rng, s).unsqueeze(0).to(DEV)
        smap = torch.full((1, 1, 512, 512), s, device=DEV)
        pred = model(noisy, smap)
        loss = charbonnier(pred, src_t) + 0.1 * (1 - M.torch_ssim(pred, src_t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] loss={loss.item():.4f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            del pred, loss, noisy; torch.cuda.empty_cache()
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(args.out, "denoise_best.pt"))


if __name__ == "__main__":
    main()