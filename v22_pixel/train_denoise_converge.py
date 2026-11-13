# -*- coding: utf-8 -*-
"""V22 纯噪点→原像素 收敛驱动（同构上色，不改架构）。
主人令：把原图转换成『真正的毫无意义的噪点』——输入是与原图完全独立的纯高斯随机噪点
（线性光空间，零原图信息），靶=原图原像素。复用 PixelEngineV22(K512字典+mode-seeking CE
+ABCDEF+梯度检查点)，抱紧 mode-seeking CE（纯L1会磨皮）。从上色收敛存点初始化。
断点续训＋逐段加轮数＋余弦退火＋平台检测，供自动任务每轮 --add_epochs 续训。
收敛靶：color 平均 CE ≲ 0.5；考试 PSNR/SSIM/ΔE 作验收。"""
import os, sys, time, json, math, argparse
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
RES = V.RES


def pure_noise(rng, sigma):
    """真正的毫无意义的噪点：与原图完全独立的纯高斯随机场（线性光），不含任何原图信息。
    中心 0.5（中灰）+ sigma 扩散 → 电视雪花状静电噪点；clip[0,1] 后转 sRGB。返回 [3,H,W]。"""
    lin = rng.normal(0.5, sigma, (3, RES, RES)).astype(np.float32)
    lin = np.clip(lin, 0.0, 1.0)
    return M.torch_linear_to_srgb(torch.tensor(lin, dtype=torch.float32))


def metrics(m, src, src_t, noisy, rule):
    m.eval()
    with torch.no_grad():
        attr = m.encode_attr([(noisy[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]])
        pred, logits, *_ = m(noisy, attr, rule)
        sf = src_t.permute(0, 2, 3, 1).reshape(-1, 3); cb = m.color_codebook
        d = (sf * sf).sum(1, keepdim=True) + (cb * cb).sum(1)[None, :] - 2 * sf @ cb.t()
        ti = d.argmin(1).reshape(1, RES, RES)
        ce = F.cross_entropy(logits, ti).item()
        pn = pred[0].permute(1, 2, 0).cpu().numpy()
        mse = float(((pn - src) ** 2).mean()); psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        ss = float(cs.ssim(pn, src))
        de = float(cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(src)).mean())
    m.train()
    return ce, psnr, ss, de, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add_epochs", type=int, default=500)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_colors", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--init", type=str, default=os.path.join(HERE, "pixel_converge", "model.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "pixel_denoise_converge"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "render"), exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    torch.cuda.set_per_process_memory_fraction(1.0)
    src = V.load_src(SRC); src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)

    state_path = os.path.join(args.out, "converge_state.json")
    ckpt_path = os.path.join(args.out, "model.pt")
    state = {"total_epochs": 0, "ce_history": [], "psnr_history": [], "de_history": [],
             "plateau": False, "best_ce": 99.0, "best_psnr": 0.0}
    if os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))

    m = T.PixelEngineV22(dim=args.dim, n_colors=args.n_colors).to(DEV)
    master = T.MasterGate().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
        m.load_state_dict(ck["model"]); master.load_state_dict(ck["master"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        print(f"[续训] 从 E{state['total_epochs']} 续，best_ce={state['best_ce']:.3f} best_psnr={state['best_psnr']:.2f}", flush=True)
    elif os.path.exists(args.init):
        ck = torch.load(args.init, map_location=DEV, weights_only=True)
        m.load_state_dict(ck["model"]); master.load_state_dict(ck["master"])
        print(f"[init] 从上色收敛存点初始化 {args.init}", flush=True)

    sigmas = [0.3, 0.4, 0.5]
    # 固定考试噪点（可复现，与原图独立）
    exam_noisy = [(s, pure_noise(np.random.default_rng(100 + i), s).unsqueeze(0).to(DEV))
                  for i, s in enumerate(sigmas)]

    def exam(tag):
        rows = []
        for s, noisy in exam_noisy:
            rule = torch.zeros(1, 16, device=DEV); rule[0, 8] = 1; rule[0, 9] = s
            ce, psnr, ss, de, pred = metrics(m, src, src_t, noisy, rule)
            rows.append((s, ce, psnr, ss, de, pred, noisy))
        print(f"  [{tag}] " + " | ".join(f"σ{s}:CE={ce:.3f}/PSNR={p:.2f}/SSIM={ss:.3f}/ΔE={de:.2f}"
                                         for s, ce, p, ss, de, *_ in rows), flush=True)
        if tag.startswith("E"):
            for s, ce, p, ss, de, pred, noisy in rows:
                trip = torch.cat([noisy[0].cpu(), pred[0].cpu(), src_t[0].cpu()], dim=2).permute(1, 2, 0).numpy()
                big = (cs.floyd_steinberg(np.clip(trip, 0, 1)) * 255).round().astype(np.uint8)
                cs.save_png_with_icc(os.path.join(args.out, "render", f"{tag}_s{s}.png"), big)
        return rows

    print(f"[纯噪点-CE] 参数 {sum(p.numel() for p in m.parameters()):,} | 输入=独立纯高斯噪点 | 靶原像素 | mode-seeking CE+0.1SSIM", flush=True)
    exam("base")
    start = state["total_epochs"]; t0 = time.time()
    for ep in range(1, args.add_epochs + 1):
        # 段内余弦退火（只降不升，单优化器保动量）——上色已验证能破平台
        lr_now = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * ep / args.add_epochs)))
        for pg in opt.param_groups: pg['lr'] = lr_now
        s = sigmas[(ep - 1) % len(sigmas)]
        noisy = pure_noise(rng, s).unsqueeze(0).to(DEV)
        rule = torch.zeros(1, 16, device=DEV); rule[0, 8] = 1; rule[0, 9] = s
        attr = m.encode_attr([(noisy[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]])
        pred, logits, ct, ac, st, gate = m(noisy, attr, rule)
        with torch.no_grad():
            sf = src_t.permute(0, 2, 3, 1).reshape(-1, 3); cb = m.color_codebook
            d = (sf * sf).sum(1, keepdim=True) + (cb * cb).sum(1)[None, :] - 2 * sf @ cb.t()
            ti = d.argmin(1).reshape(1, RES, RES)
        main_loss = F.cross_entropy(logits, ti)            # 抱紧 mode-seeking CE
        loss_B = 1.0 - M.torch_ssim(pred, src_t)           # 小 SSIM 保结构锐度
        batch = main_loss + 0.1 * loss_B
        opt.zero_grad(); batch.backward()
        if not any(p.grad is not None and torch.isnan(p.grad).any() for p in m.parameters()):
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        else: opt.zero_grad()
        if ep % 100 == 0:
            rows = exam(f"E{start + ep}")
            ce = float(np.mean([r[1] for r in rows])); psnr = float(np.mean([r[2] for r in rows]))
            de = float(np.mean([r[4] for r in rows]))
            state["ce_history"].append([start + ep, round(ce, 4)])
            state["psnr_history"].append([start + ep, round(psnr, 3)])
            state["de_history"].append([start + ep, round(de, 3)])
            state["best_ce"] = min(state["best_ce"], ce)
            state["best_psnr"] = max(state["best_psnr"], psnr)
            print(f"[E{start + ep}] 平均CE={ce:.3f} PSNR={psnr:.2f} ΔE={de:.2f} {time.time() - t0:.0f}s "
                  f"VRAM{torch.cuda.max_memory_allocated() / 2**30:.2f}GB", flush=True)
            del pred, logits, ct, ac, st, gate, batch, main_loss, loss_B, noisy
            torch.cuda.empty_cache()
    state["total_epochs"] = start + args.add_epochs
    h = [c for _, c in state["ce_history"][-4:]]
    state["plateau"] = len(h) >= 3 and (max(h) - min(h)) < 0.05
    json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    torch.save({"model": m.state_dict(), "master": master.state_dict(), "opt": opt.state_dict()}, ckpt_path)
    print(f"[段完成] 总E{state['total_epochs']} 当前CE={h[-1] if h else '?'} best_ce={state['best_ce']:.3f} "
          f"best_psnr={state['best_psnr']:.2f} 平台={state['plateau']}", flush=True)
    print(f"[收敛靶] CE≲0.5；当前{h[-1] if h else '?'}；平台期查GitHub V22+经验库", flush=True)


if __name__ == "__main__":
    main()