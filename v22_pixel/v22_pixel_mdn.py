# -*- coding: utf-8 -*-
"""V22 像素引擎 —— 主人文本字符体系的连续化（零量化）。
主人令：① 取消一切量化 ② 像素=字符，单对单对齐不平均 ③ ABCDEF 套层。
文本 CE 天然离散（字是离散 token），像素值连续 ⇒ 用 CE 的连续版＝混合密度头(MDN)：
  输入  : 连续 RGB → conv（无 to_bins、无查表量化）
  主干  : ABCDEF 六阶套层（照 GitHub V22 ABC_StageA/B/C + TurboMeta 门控残差）
  输出  : 每像素 K 个连续 RGB 模式(均值连续) + 权重 + 尺度；loss=目标色混合似然；
          推理取权重最大模式的均值＝"押定单个模式"（mode-seeking，单对单，非回归均值），全程无量化。
网络重建干净源；构造性规则（颜色组 remap + 几何 gather）照旧后施加。单组、不准 OOM。"""
import os, sys, time, math, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs

DEV = M.DEV
RES = V.RES


class ABCStage(nn.Module):
    """ABCDEF 套层的一阶（照主人 ABC_StageA/B/C 模式）：局部/空洞上下文 + 属性融合 + TurboMeta 门控残差。"""
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.ctx = nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, dim)
        self.fuse = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1), nn.GELU())
        self.gate_bias = nn.Parameter(torch.zeros(dim))
        for m in self.fuse:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)

    def forward(self, h, h_id):
        c = self.norm(self.ctx(h))
        a = h_id[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        f = self.fuse(torch.cat([c, a], 1))
        g = torch.sigmoid((self.gate_bias[None, :, None, None] + f) / 10.0)   # TurboMeta 门
        return h + g * f


class PixelEngineMDN(nn.Module):
    """连续像素 + ABCDEF 套层 + MDN 连续模式头（零量化、mode-seeking 单对单）。"""
    def __init__(self, dim=128, n_modes=5, attr_dim=256, rule_dim=16):
        super().__init__()
        self.dim = dim
        self.n_modes = n_modes
        self.p3vis = M.P3Vis(chroma=True)
        self.in_conv = nn.Conv2d(3, dim, 3, padding=1)      # 连续 RGB 输入（无量化）
        self.stages = nn.ModuleList([ABCStage(dim, d) for d in (1, 1, 1, 2, 4, 8)])
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.rule_proj = nn.Linear(rule_dim, dim)
        self.p7_gate = M.vMFGate(dim * 2, dim)
        self.inject_gain = nn.Parameter(torch.tensor(0.01))
        self.id_proj = nn.Linear(dim * 2, dim)
        self.h_scale = nn.Parameter(torch.tensor(2.0))
        # MDN 头：每像素 3K 均值 + K 对数权重 + K 对数尺度（均值连续，非 bin）
        self.head = nn.Conv2d(dim, 5 * n_modes, 1)

    def encode_attr(self, img_bgr_list):
        attrs = []
        for im in img_bgr_list:
            a = self.p3vis.process_image(im)
            attrs.append(torch.tensor(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32))
        return torch.stack(attrs).to(DEV)

    def forward(self, x, attr, rule):
        B, _, H, W = x.shape
        h = self.in_conv(x)                                 # 连续输入
        a_emb = F.normalize(self.attr_proj(attr), dim=-1)
        r_emb = F.normalize(self.rule_proj(rule), dim=-1)
        cond = torch.cat([a_emb, r_emb], dim=-1)
        gate = self.p7_gate(cond)
        h_id = self.h_scale * F.normalize(self.id_proj(cond), dim=-1)
        for stage in self.stages:                           # ABCDEF 套层
            h = stage(h, h_id)
        h = h * (1.0 + self.inject_gain * gate[:, :, None, None])
        out = self.head(h)                                  # [B,5K,H,W]
        K = self.n_modes
        means = torch.sigmoid(out[:, :3 * K]).reshape(B, K, 3, H, W)   # 连续 RGB 模式
        log_w = out[:, 3 * K:4 * K].reshape(B, K, H, W)
        log_s = out[:, 4 * K:5 * K].reshape(B, K, H, W)
        return means, log_w, log_s

    def decode(self, means, log_w, log_s):
        """取权重最大模式的均值（押定单个连续模式，非均值平均）"""
        k = log_w.argmax(1)                                 # [B,H,W]
        B, K, _, H, W = means.shape
        idx = k[:, None, None, :, :].expand(B, 1, 3, H, W)
        return means.gather(1, idx).squeeze(1)              # [B,3,H,W]


def mdn_loss(means, log_w, log_s, target, sigma_min=0.05, sigma_max=1.0):
    """混合密度负对数似然（连续 mode-seeking，零量化）。means[B,K,3,H,W] target[B,3,H,W]。"""
    sig = log_s.exp().clamp(sigma_min, sigma_max)
    log_sig = sig.log()                                     # 与二次项同一 clamp 后 sigma，防 log_s 漂负失控
    t = target[:, None, :, :, :]                            # [B,1,3,H,W]
    lp = -0.5 * ((t - means) / sig[:, :, None, :, :]) ** 2 - log_sig[:, :, None, :, :] - 0.5 * math.log(2 * math.pi)
    lp = lp.sum(2)                                          # 三通道联合 → [B,K,H,W]
    log_pi = F.log_softmax(log_w, dim=1)
    log_mix = torch.logsumexp(log_pi + lp, dim=1)           # [B,H,W]
    return -log_mix.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_modes", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "pixel_mdn_out")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    _free_b, _tot_b = torch.cuda.mem_get_info()
    _cap_gb = min(9.5, _free_b / 2**30 - 0.3)
    torch.cuda.set_per_process_memory_fraction(min(1.0, _cap_gb / (_tot_b / 2**30)))
    src = V.load_src(args.src)
    src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)   # 连续靶（无量化）
    model = PixelEngineMDN(dim=args.dim, n_modes=args.n_modes).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[mdn] 参数 {sum(p.numel() for p in model.parameters()):,} | n_modes={args.n_modes} | 零量化 | 源 512", flush=True)
    tasks = ["color", "shift", "flip", "combo", "denoise"]
    exam_pairs = [(t, V.make_pair(src, t, np.random.default_rng(43))) for t in tasks]

    def exam(tag):
        model.eval(); res = []
        with torch.no_grad():
            for t, (inp, rule, tgt) in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                means, log_w, log_s = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                pred_src = model.decode(means, log_w, log_s)
                pred = V.apply_rule(pred_src, rule.unsqueeze(0).to(DEV))[0]
                tg = tgt.to(DEV)
                l1 = float((pred - tg).abs().mean())
                p_np = pred.permute(1, 2, 0).cpu().numpy(); t_np = tg.permute(1, 2, 0).cpu().numpy()
                de_arr = cs.delta_e_2000(cs.rgb_to_lab(p_np), cs.rgb_to_lab(t_np))
                de = float(de_arr.mean()); aligned = float((de_arr < 2).mean())
                ss = cs.ssim(p_np, t_np)
                sp = float(((p_np.max(-1) - p_np.min(-1)) / (p_np.max(-1) + 1e-6)).mean())
                st = float(((t_np.max(-1) - t_np.min(-1)) / (t_np.max(-1) + 1e-6)).mean())
                res.append((t, l1, de, ss, sp, st, aligned, pred, tg, inp))
        model.train()
        print(f"  [mdn {tag}] " + " | ".join(
            f"{t}:L1={l1:.3f}/ΔE={de:.1f}/对齐{al*100:.0f}%/S={sp:.3f}->{st:.3f}"
            for t, l1, de, ss, sp, st, al, *_ in res), flush=True)
        if tag.startswith("E"):
            for t, l1, de, ss, sp, st, al, pred, tg, inp in res:
                trip = torch.cat([inp, pred.cpu(), tg.cpu()], dim=2).permute(1, 2, 0).numpy()
                trip = cs.floyd_steinberg(np.clip(trip, 0, 1))
                big = (trip * 255).round().astype(np.uint8)
                cs.save_png_with_icc(os.path.join(out, f"{tag}_{t}.png"), big)
        return res

    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        inp, rule, tgt = V.make_pair(src, tasks[(ep - 1) % len(tasks)], rng)
        inp = inp.unsqueeze(0).to(DEV); rule = rule.unsqueeze(0).to(DEV)
        attr = model.encode_attr([(inp[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]])
        means, log_w, log_s = model(inp, attr, rule)
        sig_min = 0.3 * (0.05 / 0.3) ** (ep / args.epochs)  # sigma 退火 0.3→0.05：先粗训防梯度消失，后精细
        loss = mdn_loss(means, log_w, log_s, src_t, sigma_min=sig_min)   # 连续混合似然（零量化）
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] NLL={loss.item():.4f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "pixel_mdn_best.pt"))


if __name__ == "__main__":
    main()