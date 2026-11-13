# -*- coding: utf-8 -*-
"""V22 像素 token 引擎 —— 严格照主人文本字符处理（V21 CharEmbed→P6 tied CE / img2img I2IGen）。
把每个像素色值当成文本里的一个字符：
  输入  : 色值 → to_bins → ColorEmbed 查表（正交初始化，对应 CharEmbed/bin_embed）
  输出  : hidden × ColorEmbed^T → vocab logits → CE 直出"色字"（权重绑定，对应 P6 word_vec×char_embed^T）
  解码  : argmax → (bin+0.5)/n_bins（对应 decode_bins）
网络重建干净源（CE 逼每像素押定确切色 token，mode-seeking，单对单对齐，非回归均值）；
构造性规则（颜色组 remap + 几何 gather）照旧在网络输出后施加。
单组课程、不准 OOM。忘掉连续回归，按主人字符体系做。"""
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


def to_bins(rgb, n_bins):
    """色值∈[0,1] → bin id（主人 to_bins: rgb*n_bins → int64 clamp）"""
    return (rgb * n_bins).long().clamp(0, n_bins - 1)


def decode_bins(bins, n_bins):
    """bin id → 色值（主人 decode_bins: (bin+0.5)/n_bins）"""
    return (bins.float() + 0.5) / n_bins


class ABCStage(nn.Module):
    """ABCDEF 套层的一阶（照主人 ABC_StageA/B/C 模式）：局部/空洞上下文 + 属性融合 + TurboMeta 门控残差。
       ctx=邻域联动(P3-L 像素版)；fuse=concat(上下文,全局身份)→MLP(StageB fuse)；gate=sigmoid((bias+x)/10)(TurboMeta)。"""
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
        c = self.norm(self.ctx(h))                                   # 邻域/空洞上下文
        a = h_id[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
        f = self.fuse(torch.cat([c, a], 1))                          # 融合上下文+全局身份
        g = torch.sigmoid((self.gate_bias[None, :, None, None] + f) / 10.0)   # TurboMeta 门
        return h + g * f                                             # 门控残差


class PixelTokenEngine(nn.Module):
    """像素=字符。ColorEmbed 字典（正交、权重绑定）+ 查表输入 + tied CE 输出。"""
    def __init__(self, dim=128, n_bins=64, attr_dim=256, rule_dim=16):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        # 颜色字典 = CharEmbed：n_bins 个色字 × dim，正交初始化，每色独立方向；输入查表+输出分类共用（tied）
        self.color_embed = nn.Parameter(torch.empty(n_bins, dim))
        nn.init.orthogonal_(self.color_embed)
        self.p3vis = M.P3Vis(chroma=True)
        # ABCDEF 套层（照 GitHub V22 ABC_StageA/B/C+Gate 模式）：6 阶融合+门控残差，后阶空洞扩上下文
        self.stages = nn.ModuleList([ABCStage(dim, d) for d in (1, 1, 1, 2, 4, 8)])
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.rule_proj = nn.Linear(rule_dim, dim)
        self.p7_gate = M.vMFGate(dim * 2, dim)            # P7 全局 vMF 球面联动
        self.inject_gain = nn.Parameter(torch.tensor(0.01))
        self.id_proj = nn.Linear(dim * 2, dim)            # 全局身份（VisualEmbed 对应）
        self.h_scale = nn.Parameter(torch.tensor(2.0))    # 球面归一化后可学习尺度（v4.1）
        # 输出头：每像素 3 通道各一个隐状态 → 对字典分类（P6 tied CE）
        self.head = nn.Conv2d(dim, 3 * dim, 1)

    def encode_attr(self, img_bgr_list):
        attrs = []
        for im in img_bgr_list:
            a = self.p3vis.process_image(im)
            attrs.append(torch.tensor(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32))
        return torch.stack(attrs).to(DEV)

    def forward(self, x, attr, rule):
        B, _, H, W = x.shape
        # 输入：色值 → 色字查表（每像素 3 通道字符嵌入相加）
        bins = to_bins(x, self.n_bins)                    # [B,3,H,W]
        emb = self.color_embed[bins]                      # [B,3,H,W,dim]
        h = emb.sum(1).permute(0, 3, 1, 2).contiguous()   # [B,dim,H,W]
        a_emb = F.normalize(self.attr_proj(attr), dim=-1)
        r_emb = F.normalize(self.rule_proj(rule), dim=-1)
        cond = torch.cat([a_emb, r_emb], dim=-1)
        gate = self.p7_gate(cond)                         # [B,dim] P7 全局 vMF
        h_id = self.h_scale * F.normalize(self.id_proj(cond), dim=-1)   # 全局身份
        for stage in self.stages:                         # ABCDEF 套层（融合+门控残差）
            h = stage(h, h_id)
        h = h * (1.0 + self.inject_gain * gate[:, :, None, None])       # Gate 调制 P6（主人 Gate→P6）
        # 输出：hidden × ColorEmbed^T → 色字 logits（tied weights，CE 直出色字）
        hidden = self.head(h).reshape(B, 3, self.dim, H, W)
        logits = torch.einsum('bcdhw,kd->bchwk', hidden, self.color_embed)   # [B,3,H,W,n_bins]
        return logits

    def decode(self, logits):
        return decode_bins(logits.argmax(-1), self.n_bins)   # [B,3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_bins", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "pixel_token_out")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    _free_b, _tot_b = torch.cuda.mem_get_info()
    _cap_gb = min(9.5, _free_b / 2**30 - 0.3)             # 照原件 img2img：尊重当前空闲，留余量防溢共享
    torch.cuda.set_per_process_memory_fraction(min(1.0, _cap_gb / (_tot_b / 2**30)))
    src = V.load_src(args.src)
    src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)
    src_bins = to_bins(src_t, args.n_bins)                 # CE 靶 = 干净源色字（网络重建源，规则后施加）
    model = PixelTokenEngine(dim=args.dim, n_bins=args.n_bins).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[token] 参数 {sum(p.numel() for p in model.parameters()):,} | n_bins={args.n_bins} | 源 512", flush=True)
    tasks = ["color", "shift", "flip", "combo", "denoise"]
    exam_pairs = [(t, V.make_pair(src, t, np.random.default_rng(43))) for t in tasks]

    def exam(tag):
        model.eval(); res = []
        with torch.no_grad():
            for t, (inp, rule, tgt) in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                logits = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                pred_src = model.decode(logits)
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
        print(f"  [token {tag}] " + " | ".join(
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
        logits = model(inp, attr, rule)
        loss = F.cross_entropy(logits.reshape(-1, args.n_bins), src_bins.reshape(-1))   # CE 直出色字
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] CE={loss.item():.4f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "pixel_token_best.pt"))


if __name__ == "__main__":
    main()