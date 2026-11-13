# -*- coding: utf-8 -*-
"""V22 噪点→原像素（同构上色，不改架构）：复用 PixelEngineV22(K512字典+mode-seeking CE+ABCDEF+检查点)。
主人令：抱紧 mode-seeking CE（纯L1去噪会磨皮），输入噪点图(线性光高斯+椒盐+块打乱)，靶=原图原像素。
从上色收敛存点初始化(模型已认识该图像素，去噪收敛更快)。E1000。"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs
DEV = T.DEV
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"


def add_noise_linear(clean_t, rng, sigma):
    """线性光空间加噪：高斯+椒盐+块打乱（主人文档物理规矩）。clean_t [3,H,W]∈[0,1] sRGB。"""
    lin = M.srgb_to_linear(clean_t).cpu().numpy()
    g = rng.normal(0, sigma, lin.shape).astype(np.float32)
    noisy = lin + g
    # 椒盐 2%
    mask = rng.random(lin.shape[1:]) < 0.02
    salt = rng.random(lin.shape[1:]) < 0.5
    noisy[:, mask] = np.where(salt[mask], 1.0, 0.0)
    # 块打乱：随机 8×8 块置局部均值（dropout）
    for _ in range(6):
        y = int(rng.integers(0, 512 - 8)); x = int(rng.integers(0, 512 - 8))
        noisy[:, y:y+8, x:x+8] = noisy[:, y:y+8, x:x+8].mean(axis=(1, 2), keepdims=True)
    noisy = np.clip(noisy, 0, 1)
    return M.torch_linear_to_srgb(torch.tensor(noisy, dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_colors", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--init", type=str, default=os.path.join(HERE, "pixel_converge", "model.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "pixel_denoise_ce"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    torch.cuda.set_per_process_memory_fraction(1.0)
    src = V.load_src(SRC); src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)
    m = T.PixelEngineV22(dim=args.dim, n_colors=args.n_colors).to(DEV)
    master = T.MasterGate().to(DEV)
    if os.path.exists(args.init):
        ck = torch.load(args.init, map_location=DEV, weights_only=True)
        m.load_state_dict(ck["model"]); master.load_state_dict(ck["master"])
        print(f"[init] 从上色收敛存点初始化 {args.init}", flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    opt_master = torch.optim.Adam(master.parameters(), lr=args.lr * 0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    print(f"[denoise-CE] 参数 {sum(p.numel() for p in m.parameters()):,} | mode-seeking CE+SSIM | 靶原像素", flush=True)
    sigmas = [0.05, 0.1, 0.15, 0.2]
    # 固定考试噪点（可复现）
    exam_noisy = [(s, add_noise_linear(src_t[0], np.random.default_rng(43), s).unsqueeze(0).to(DEV)) for s in sigmas]

    def exam(tag):
        m.eval(); res = []
        with torch.no_grad():
            for s, noisy in exam_noisy:
                rule = torch.zeros(1, 16, device=DEV); rule[0, 8] = 1; rule[0, 9] = s
                attr = m.encode_attr([(noisy[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
                pred, logits, *_ = m(noisy, attr, rule)
                pn = pred[0].permute(1,2,0).cpu().numpy()
                mse = float(((pn - src) ** 2).mean()); psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
                ss = cs.ssim(pn, src); de = float(cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(src)).mean())
                res.append((s, psnr, ss, de, pred, noisy))
        m.train()
        print(f"  [denoise {tag}] " + " | ".join(f"σ{s}:PSNR={p:.2f}/SSIM={ss:.3f}/ΔE={de:.2f}" for s, p, ss, de, *_ in res), flush=True)
        if tag.startswith("E"):
            for s, p, ss, de, pred, noisy in res:
                trip = torch.cat([noisy[0].cpu(), pred[0].cpu(), src_t[0].cpu()], dim=2).permute(1,2,0).numpy()
                big = (cs.floyd_steinberg(np.clip(trip, 0, 1)) * 255).round().astype(np.uint8)
                cs.save_png_with_icc(os.path.join(args.out, f"{tag}_s{s}.png"), big)
        return res

    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        m.train()
        s = sigmas[(ep - 1) % len(sigmas)]
        noisy = add_noise_linear(src_t[0], rng, s).unsqueeze(0).to(DEV)
        rule = torch.zeros(1, 16, device=DEV); rule[0, 8] = 1; rule[0, 9] = s
        attr = m.encode_attr([(noisy[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
        pred, logits, ct, ac, st, gate = m(noisy, attr, rule)
        with torch.no_grad():
            sf = src_t.permute(0,2,3,1).reshape(-1,3); cb = m.color_codebook
            d = (sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t(); ti = d.argmin(1).reshape(1,512,512)
        main_loss = F.cross_entropy(logits, ti)                 # 抱紧 mode-seeking CE
        loss_B = 1.0 - M.torch_ssim(pred, src_t)                # 小 SSIM 保结构锐度
        batch = main_loss + 0.1 * loss_B
        opt.zero_grad(); opt_master.zero_grad(); batch.backward()
        if not any(p.grad is not None and torch.isnan(p.grad).any() for p in m.parameters()):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt_master.step()
        else: opt.zero_grad(); opt_master.zero_grad()
        sched.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] CE={main_loss.item():.4f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            del pred, logits, batch, main_loss, noisy; torch.cuda.empty_cache()
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save({"model": m.state_dict(), "master": master.state_dict()}, os.path.join(args.out, "denoise_ce_best.pt"))


if __name__ == "__main__":
    main()