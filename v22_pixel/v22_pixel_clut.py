# -*- coding: utf-8 -*-
"""V22 像素级图生图 · 纯离散 CLUT 架构（2026-11-12，主人令：不平滑，直接用原本像素颜色+位置编码组）

根因修正：连续重建+梯度域/CCV平滑 在 sRGB gamma 空间取均值会压暗（传统色彩科学：
  sRGB 黑白平均=0.5编码≈0.214线性光）。故弃平滑，改纯离散索引色：
  · 图像 = 位置编码组（索引图）+ 调色板（CLUT，颜色组→精确RGB）。
  · 颜色规则 = remap 调色板条目（颜色组），中性背景组恒等；索引图(位置)不动。
  · 几何规则 = 索引图的位置 gather（平移/翻转/仿射）。
  · 网络只预测"每个位置属于哪个颜色组"（K-way 分类）+ 内容重建；颜色取调色板精确值。
  · 输出 = remapped_palette[index]，离散精确，无连续平均 ⇒ 无 gamma 暗化、无斑点。
  · loss = 颜色组交叉熵（离散，不平滑）+ 位置组一致。
  · P7 联动 = vMF 球面门控（主人最新球面件）；P3Vis 图像知识；万物皆Token球面Embed。
理论根据：索引色/CLUT + 颜色聚合向量CCV(Pass1996) + Morton位置编码 + Porter-Duff合成。
硬约束：单组、不准OOM、原件只读。
"""
import os, sys, glob, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from P3_vision.p3vis import P3Vis
from v22_pixel_clean import (vMFGate, geo_gather, color_group_remap, COLOR_MULT,
                             imread_u8, imwrite_u8, load_rgb, DEV, IMG, DATA_DIR)

OUT_DIR = os.path.join(HERE, "pixel_clut_out")
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════ CLUT 调色板（颜色组） ════════════════════════
def build_palette(src, K):
    """全局统一网格调色板（K 级/通道 → K³ 颜色组，所有源共享 ⇒ 组索引跨源通用）。
       返回 (palette[K³,3], source_index[H,W])。索引图=每位置颜色组编号（位置编码组）。"""
    levels = (torch.arange(K, dtype=torch.float32) + 0.5) / K
    palette = torch.cartesian_prod(levels, levels, levels)   # [K³,3] 全局固定调色板
    q = (torch.tensor(src.reshape(-1, 3), dtype=torch.float32) * K).long().clamp(0, K - 1)
    gid = q[:, 0] * K * K + q[:, 1] * K + q[:, 2]
    index = gid.reshape(IMG, IMG)
    return palette, index


def remap_palette(palette, mult):
    """颜色规则 remap 调色板条目 [K,3]（颜色组）。中性组(背景)恒等。返回 [K,3]。"""
    mult = torch.as_tensor(mult, dtype=palette.dtype, device=palette.device).reshape(1, 3)
    sat = palette.max(1, keepdim=True).values - palette.min(1, keepdim=True).values   # [K,1]
    w = (sat * 4.0).clamp(0.0, 1.0)
    return (palette * (1.0 + w * (mult - 1.0))).clamp(0.0, 1.0)


# ════════════════════════ 离散 CLUT 引擎 ════════════════════════
class CLUTEngine(nn.Module):
    """CNN 像素基底 + P3知识 + vMF球面联动 → 每位置颜色组 logits（K-way）。
       颜色取调色板精确值，不平滑。"""
    def __init__(self, n_groups, dim=128, attr_dim=256, rule_dim=16):
        super().__init__()
        self.p3vis = P3Vis()
        self.enc = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.rule_proj = nn.Linear(rule_dim, dim)
        self.p7_gate = vMFGate(dim * 2, dim)
        self.inject_gain = nn.Parameter(torch.tensor(0.01))
        self.head = nn.Conv2d(dim, n_groups, 3, padding=1)   # 颜色组分类头

    def encode_attr(self, img_bgr_list):
        return torch.stack([torch.tensor(self.p3vis.process_image(im), dtype=torch.float32)
                            for im in img_bgr_list]).to(DEV)

    def forward(self, x, attr, rule):
        h = self.enc(x)
        a = F.normalize(self.attr_proj(attr), dim=-1)
        r = F.normalize(self.rule_proj(rule), dim=-1)
        gate = self.p7_gate(torch.cat([a, r], dim=-1))
        h = h + self.inject_gain * gate.unsqueeze(-1).unsqueeze(-1) * h
        return self.head(h)                                  # [B,K,H,W] 颜色组 logits


# ════════════════════════ 数据 ════════════════════════
def make_pair(src, src_index, task, rng, K):
    src_t = torch.tensor(src, dtype=torch.float32).permute(2, 0, 1)
    rule = np.zeros(16, dtype=np.float32)
    mult = None
    geo = None
    if task == "color":
        name = rng.choice(list(COLOR_MULT)); mult = COLOR_MULT[name]
        rule[0] = 1.0; rule[1:4] = mult
        gray = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        inp = np.stack([gray, gray, gray], -1)
    elif task == "shift":
        dy, dx = int(rng.integers(2, 15)), int(rng.integers(2, 15))
        rule[4] = 1.0; rule[5] = dy / IMG; rule[6] = dx / IMG
        inp = src.copy(); geo = ("shift", (dy, dx))
    elif task == "flip":
        rule[7] = 1.0; inp = src.copy(); geo = ("flip", None)
    elif task == "combo":
        name = rng.choice(["偏红", "暖调", "冷调"]); mult = COLOR_MULT[name]
        dy, dx = int(rng.integers(2, 15)), int(rng.integers(2, 15))
        rule[0] = 1.0; rule[1:4] = mult; rule[4] = 1.0; rule[5] = dy / IMG; rule[6] = dx / IMG
        gray = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        inp = np.stack([gray, gray, gray], -1); geo = ("shift", (dy, dx))
    else:
        sigma = float(rng.choice([0.1, 0.2, 0.3]))
        rule[8] = 1.0; rule[9] = sigma
        inp = np.clip(src + rng.normal(0, sigma, src.shape).astype(np.float32), 0, 1)
    inp_t = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1)
    return inp_t, torch.tensor(rule), mult, geo


def render_clut(idx, palette, mult, geo):
    """idx [B,H,W] 颜色组索引 → 精确调色板颜色（remap颜色组）→ 几何gather。离散无平滑。"""
    pal = palette
    if mult is not None:
        pal = remap_palette(palette, mult)
    out = pal[idx]                                           # [B,H,W,3] 精确颜色
    out = out.permute(0, 3, 1, 2)                            # [B,3,H,W]
    if geo is not None:
        out = geo_gather(out, geo[0], geo[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--K", type=int, default=16, help="每通道量化级（颜色组粒度）")
    ap.add_argument("--exam_every", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6.0 / (torch.cuda.get_device_properties(0).total_memory / 2**30)))

    files = {os.path.basename(f).rsplit(".", 1)[0]: f for f in glob.glob(os.path.join(DATA_DIR, "*.bmp"))}
    train_src = load_rgb(files["apple_01"])
    palette, src_index = build_palette(train_src, args.K)
    n_groups = len(palette)
    palette = palette.to(DEV); src_index = src_index.to(DEV)
    print(f"[CLUT] 调色板 {n_groups} 颜色组 | 单组训练源 apple_01", flush=True)

    exam_names = ["apple_01", "apple_16", "apple_17", "apple_18", "apple_19", "apple_20", "sun_05"]
    exam_src = {n: load_rgb(files[n]) for n in exam_names if n in files}
    model = CLUTEngine(n_groups, dim=args.dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[模型] 参数 {sum(p.numel() for p in model.parameters()):,}", flush=True)

    tasks = ["color", "shift", "flip", "combo", "denoise"]
    # 考试用各源自己的调色板索引（颜色组），目标=该源精确颜色remap+几何
    exam_pairs = []
    for nm, s in exam_src.items():
        pal_e, idx_e = build_palette(s, args.K)
        for t in tasks:
            inp, rule, mult, geo = make_pair(s, idx_e, t, np.random.default_rng(args.seed + 1), args.K)
            tgt = render_clut(idx_e.unsqueeze(0), pal_e.to(DEV), mult, geo)[0]
            exam_pairs.append((nm, t, inp, rule, mult, geo, tgt, pal_e.to(DEV), idx_e.to(DEV)))

    def exam(tag):
        model.eval(); rows = []
        with torch.no_grad():
            for nm, t, inp, rule, mult, geo, tgt, pal_e, idx_e in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                logits = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                pred = render_clut(logits.argmax(1), pal_e, mult, geo)[0]
                tg = tgt.to(DEV)
                l1 = float((pred - tg).abs().mean())
                pc = pred.reshape(-1).cpu().numpy(); tc = tg.reshape(-1).cpu().numpy()
                pear = float(np.corrcoef(pc, tc)[0, 1]) if pc.std() > 1e-8 else float("nan")
                rows.append((nm, t, l1, pear, pred, tg, inp))
        model.train()
        bt = {}
        for nm, t, l1, pear, *_ in rows:
            bt.setdefault(t, []).append((l1, pear))
        print(f"  [i2i {tag}] 总体 L1={np.mean([r[2] for r in rows]):.4f} 皮尔逊={np.nanmean([r[3] for r in rows]):.4f}", flush=True)
        for t, vs in bt.items():
            print(f"    {t:8s} L1={np.mean([v[0] for v in vs]):.4f} 皮尔逊={np.nanmean([v[1] for v in vs]):.4f}", flush=True)
        if tag.startswith("E"):
            for nm, t, l1, pear, pred, tg, inp in rows:
                if nm != "apple_01":
                    continue
                trip = torch.cat([inp, pred.cpu(), tg.cpu()], dim=2).permute(1, 2, 0).numpy()
                big = cv2.resize((trip * 255).round().astype(np.uint8), (IMG * 3 * 8, IMG * 8), interpolation=cv2.INTER_NEAREST)
                imwrite_u8(os.path.join(OUT_DIR, f"{tag}_{nm}_{t}.png"), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        batch = [make_pair(train_src, src_index, tasks[i % len(tasks)], rng, args.K) for i in range(args.bs)]
        inp = torch.stack([b[0] for b in batch]).to(DEV)
        rule = torch.stack([b[1] for b in batch]).to(DEV)
        attr = model.encode_attr([(b[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1] for b in batch])
        logits = model(inp, attr, rule)
        # 颜色组交叉熵（离散，不平滑）：目标=源索引图（几何变换后的索引）
        loss = 0.0
        for i, b in enumerate(batch):
            mult, geo = b[2], b[3]
            tgt_idx = src_index.clone()
            if geo is not None:
                ti = F.one_hot(tgt_idx, n_groups).permute(2, 0, 1).unsqueeze(0).float()
                ti = geo_gather(ti, geo[0], geo[1])[0].argmax(0)
                tgt_idx = ti
            loss = loss + F.cross_entropy(logits[i].unsqueeze(0), tgt_idx.unsqueeze(0))
        loss = loss / args.bs
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] loss={loss.item():.4f} gain={model.inject_gain.item():.3f} {time.time()-t0:.0f}s", flush=True)
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s | GPU峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "clut_best.pt"))


if __name__ == "__main__":
    main()