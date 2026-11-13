# -*- coding: utf-8 -*-
"""512×512 单图训练变体（复用 v22_pixel_clean 的模型/规则/loss，数据分辨率 512）。
主人给的红苹果 512×512，测大分辨率单图图生图效果。"""
import os, sys, time, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v22_pixel_clean as M
import color_science as cs

DEV = M.DEV
RES = 512
SRC_PATH = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\e44fec27-7adf-4cd3-922f-ed3ad7044cd9.png"
OUT = os.path.join(HERE, "pixel_512_out")
os.makedirs(OUT, exist_ok=True)


def load_src(path=None):
    im = cv2.imread(path or SRC_PATH)
    im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def make_pair(src, task, rng):
    src_t = torch.tensor(src).permute(2, 0, 1)
    rule = np.zeros(16, dtype=np.float32)
    mult = None; geo = None
    if task == "color":
        name = rng.choice(list(M.COLOR_MULT)); mult = M.COLOR_MULT[name]
        rule[0] = 1; rule[1:4] = mult
        g = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        inp = np.stack([g, g, g], -1)
    elif task == "shift":
        dy, dx = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        rule[4] = 1; rule[5] = dy / RES; rule[6] = dx / RES
        inp = src.copy(); geo = ("shift", (dy, dx))
    elif task == "flip":
        rule[7] = 1; inp = src.copy(); geo = ("flip", None)
    elif task == "combo":
        name = rng.choice(["偏红", "暖调", "冷调"]); mult = M.COLOR_MULT[name]
        dy, dx = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        rule[0] = 1; rule[1:4] = mult; rule[4] = 1; rule[5] = dy / RES; rule[6] = dx / RES
        g = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        inp = np.stack([g, g, g], -1); geo = ("shift", (dy, dx))
    else:
        sigma = float(rng.choice([0.05, 0.1, 0.15]))
        rule[8] = 1; rule[9] = sigma
        inp = np.clip(src + rng.normal(0, sigma, src.shape).astype(np.float32), 0, 1)
    inp_t = torch.tensor(inp).permute(2, 0, 1)
    tgt = src_t.clone()
    if mult is not None:
        tgt = M.color_group_remap(tgt.unsqueeze(0), torch.tensor(mult))[0]
    if geo is not None:
        tgt = M.geo_gather(tgt.unsqueeze(0), geo[0], geo[1])[0]
    return inp_t, torch.tensor(rule), tgt


def apply_rule(pred, rule):
    out = []
    for i in range(pred.shape[0]):
        o = pred[i]; r = rule[i]
        if r[0] > 0.5:
            o = M.color_group_remap(o.unsqueeze(0), r[1:4])[0]
        if r[4] > 0.5:
            dy, dx = int(round(float(r[5]) * RES)), int(round(float(r[6]) * RES))
            o = M.geo_gather(o.unsqueeze(0), "shift", (dy, dx))[0]
        if r[7] > 0.5:
            o = M.geo_gather(o.unsqueeze(0), "flip")[0]
        out.append(o)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = args.out
        os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6.0 / (torch.cuda.get_device_properties(0).total_memory / 2**30)))
    src = load_src(args.src)
    model = M.PixelEngine(dim=args.dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[512] 参数 {sum(p.numel() for p in model.parameters()):,} | 源 红苹果 512", flush=True)
    tasks = ["color", "shift", "flip", "combo", "denoise"]
    exam_pairs = [(t, make_pair(src, t, np.random.default_rng(43))) for t in tasks]

    def exam(tag):
        model.eval(); res = []
        with torch.no_grad():
            for t, (inp, rule, tgt) in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                ps = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                pred = apply_rule(ps, rule.unsqueeze(0).to(DEV))[0]
                tg = tgt.to(DEV)
                l1 = float((pred - tg).abs().mean())
                p_np = pred.permute(1, 2, 0).cpu().numpy(); t_np = tg.permute(1, 2, 0).cpu().numpy()
                de = float(cs.delta_e_2000(cs.rgb_to_lab(p_np), cs.rgb_to_lab(t_np)).mean())
                ss = cs.ssim(p_np, t_np)
                sp = float(((p_np.max(-1) - p_np.min(-1)) / (p_np.max(-1) + 1e-6)).mean())
                st = float(((t_np.max(-1) - t_np.min(-1)) / (t_np.max(-1) + 1e-6)).mean())
                res.append((t, l1, de, ss, pred, tg, inp, sp, st))
        model.train()
        print(f"  [512 {tag}] " + " | ".join(f"{t}:L1={l1:.3f}/ΔE={de:.1f}/SSIM={ss:.3f}/S={sp:.3f}->{st:.3f}" for t, l1, de, ss, pred, tg, inp, sp, st in res), flush=True)
        if tag.startswith("E"):
            for t, l1, de, ss, pred, tg, inp, sp, st in res:
                trip = torch.cat([inp, pred.cpu(), tg.cpu()], dim=2).permute(1, 2, 0).numpy()
                trip = cs.floyd_steinberg(np.clip(trip, 0, 1))
                big = (trip * 255).round().astype(np.uint8)
                cs.save_png_with_icc(os.path.join(OUT, f"{tag}_{t}.png"), big)
        return res

    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        inp, rule, tgt = make_pair(src, tasks[(ep - 1) % len(tasks)], rng)
        inp = inp.unsqueeze(0).to(DEV); rule = rule.unsqueeze(0).to(DEV); tgt = tgt.unsqueeze(0).to(DEV)
        attr = model.encode_attr([(inp[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]])
        ps = model(inp, attr, rule)
        pred = apply_rule(ps, rule)
        loss = M.structural_loss(pred, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] loss={loss.item():.4f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, "pixel_512_best.pt"))


if __name__ == "__main__":
    main()