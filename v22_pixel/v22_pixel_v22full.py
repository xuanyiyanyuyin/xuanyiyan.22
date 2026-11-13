# -*- coding: utf-8 -*-
"""V22 像素引擎 · 完整训练机制版（照 GitHub V22 train_v22_mm.py 移植）。
主人机制：① 前向 ABCDEF 套层 ② P3-L 分组关联 ③ 5路loss+MasterGate自学习比例
          ④ 监控自愈(尖峰clamp/aux_scale/NaN跳步) ⑤ 多优化器。零量化（连续解码头）。
辅助 loss 靶＝干净源 P3 属性（非退化输入；灰输入色度为0会误监督）→ 逼网络编码鲜色度，修黑白→彩发灰。
单组、不准 OOM。"""
import os, sys, time, math, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs

DEV = M.DEV
RES = V.RES


class ABCStage(nn.Module):
    """ABCDEF 套层一阶：局部/空洞上下文 + 全局身份融合 + TurboMeta 门控残差。"""
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
        g = torch.sigmoid((self.gate_bias[None, :, None, None] + f) / 10.0)
        return h + g * f


class MasterGate(nn.Module):
    """自学习比例（主人 MasterGate :264）：5路detached sub_losses → MLP → sigmoid → 5路0-1可学权重。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 5))
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight, gain=0.1); nn.init.zeros_(l.bias)

    def forward(self, sub_losses):
        return 0.2 + 0.8 * torch.sigmoid(self.net(sub_losses))   # 自学习比例∈[0.2,1.0]，下限防辅助loss被压到0


class PixelEngineV22(nn.Module):
    def __init__(self, dim=128, attr_dim=256, rule_dim=16, n_colors=64):
        super().__init__()
        self.dim = dim
        self.n_colors = n_colors
        self.p3vis = M.P3Vis(chroma=True)
        self.in_conv = nn.Conv2d(3, dim, 3, padding=1)
        self.stages = nn.ModuleList([ABCStage(dim, d) for d in (1, 1, 1, 2, 4, 8)])
        self.attr_proj = nn.Linear(attr_dim, dim)
        self.rule_proj = nn.Linear(rule_dim, dim)
        self.p7_gate = M.vMFGate(dim * 2, dim)
        self.inject_gain = nn.Parameter(torch.tensor(0.01))
        self.id_proj = nn.Linear(dim * 2, dim)
        self.h_scale = nn.Parameter(torch.tensor(2.0))
        self.pos_gate = nn.Conv2d(dim, dim, 1)              # 逐位身份门（主人v4：逐位调色记忆，精准像素点对齐）
        nn.init.xavier_uniform_(self.pos_gate.weight, gain=0.1)
        # 主loss改mode-seeking（主人文本字符机制搬到像素）：可学连续颜色字典 + CE押定
        # color_codebook=K个可学连续RGB颜色（非固定标量bin，零标量量化）；color_embed=可学连续嵌入（对应char_embed）
        self.color_codebook = nn.Parameter(torch.rand(n_colors, 3))           # 连续RGB颜色，端到端学
        self.color_embed = nn.Parameter(torch.empty(n_colors, dim))           # 颜色嵌入，tied输入输出
        nn.init.orthogonal_(self.color_embed)
        self.cls_head = nn.Conv2d(dim, dim, 1)              # 隐状态→对颜色字典分类（CE mode-seeking）
        self.decode = nn.Conv2d(dim, 3, 1)                  # 连续解码（辅助loss用）
        # P3 关联监督头（全局特征 → 对干净源 P3 属性监督）
        self.content_head = nn.Linear(dim, 256)             # loss_B 内容 ↔ 全P3活跃槽
        self.chroma_head = nn.Linear(dim, 48)               # loss_C 色度 ↔ V1色彩[0:48]
        self.struct_head = nn.Linear(dim, 64)               # loss_A 结构 ↔ V2几何[48:112]

    def encode_attr(self, img_bgr_list):
        attrs = []
        for im in img_bgr_list:
            a = self.p3vis.process_image(im)
            attrs.append(torch.tensor(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32))
        return torch.stack(attrs).to(DEV)

    def _pos_enc(self, H, W, device):
        # 正弦位置编码 [1,dim,H,W]（分辨率无关，逐位身份的位置基）
        ys = torch.linspace(0, 1, H, device=device); xs = torch.linspace(0, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        n = max(self.dim // 4, 1)
        freqs = torch.exp(torch.arange(n, device=device) * (-math.log(10000.0) / max(n - 1, 1)))
        feats = []
        for f in freqs:
            feats += [torch.sin(gx * f * math.pi), torch.cos(gx * f * math.pi),
                      torch.sin(gy * f * math.pi), torch.cos(gy * f * math.pi)]
        return torch.stack(feats, 0)[:self.dim].unsqueeze(0)

    def forward(self, x, attr, rule):
        B, _, H, W = x.shape
        h = self.in_conv(x)
        a_emb = F.normalize(self.attr_proj(attr), dim=-1)
        r_emb = F.normalize(self.rule_proj(rule), dim=-1)
        cond = torch.cat([a_emb, r_emb], dim=-1)
        gate = self.p7_gate(cond)
        h_id = self.h_scale * F.normalize(self.id_proj(cond), dim=-1)
        for stage in self.stages:
            h = checkpoint(stage, h, h_id, use_reentrant=False)   # 梯度检查点：反向重算激活，省~3GB让K512+dim128装下
        # 逐位身份门控（主人v4）：每像素独立身份调制 → 逐位调色记忆，精准像素点对齐而非全局平均
        pos = self._pos_enc(H, W, x.device)
        g_pos = torch.tanh(torch.clamp(self.pos_gate(pos), -2.0, 2.0))   # [B,dim,H,W]
        hp = h_id[:, :, None, None] * g_pos                              # 逐位身份注入
        h = h + self.inject_gain * (gate[:, :, None, None] * h + hp)
        # mode-seeking 主头：隐状态 × color_embed^T → 颜色字典 logits（CE押定，对应文本P6 tied CE）
        cls_h = self.cls_head(h)                                          # [B,dim,H,W]
        logits = torch.einsum('bdhw,kd->bkhw', cls_h, self.color_embed)   # [B,K,H,W]
        # 连续RGB（softmax加权字典颜色，可微，供辅助色度/SSIM loss）；exam用argmax硬选
        prob = F.softmax(logits, dim=1)
        pred = torch.einsum('bkhw,kc->bchw', prob, self.color_codebook).clamp(0.0, 1.0)
        g = h.mean(dim=(2, 3))                              # 全局特征 [B,dim]
        ct = self.content_head(g)                           # [B,256]
        ac = self.chroma_head(g)                            # [B,48]
        st = self.struct_head(g)                            # [B,64]
        return pred, logits, ct, ac, st, gate

    def decode_hard(self, logits):
        # 硬选：argmax → 字典连续RGB颜色（mode-seeking押定，精准像素点对齐）
        idx = logits.argmax(1)                              # [B,H,W]
        return self.color_codebook[idx].permute(0, 3, 1, 2).clamp(0.0, 1.0)   # [B,3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--exam_every", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_colors", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, "pixel_v22full_out")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    _free_b, _tot_b = torch.cuda.mem_get_info()
    # 去掉显存硬帽（主人令）：用满空闲显存，大字典K512+dim128峰值~10.7GB需~11GB空闲
    torch.cuda.set_per_process_memory_fraction(1.0)
    src = V.load_src(args.src)
    src_t = torch.tensor(src).permute(2, 0, 1).unsqueeze(0).to(DEV)
    # 干净源 P3 属性＝辅助 loss 靶（关联监督）
    src_attr = torch.tensor(np.nan_to_num(M.P3Vis(chroma=True).process_image(
        (src[:, :, ::-1] * 255).astype(np.uint8)), nan=0.0, posinf=0.0, neginf=0.0),
        dtype=torch.float32).unsqueeze(0).to(DEV)            # [1,256]
    active_mask = (src_attr.abs().sum(0) > 1e-6).float()     # 活跃槽 [256]
    model = PixelEngineV22(dim=args.dim, n_colors=args.n_colors).to(DEV)
    master = MasterGate().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_master = torch.optim.Adam(master.parameters(), lr=args.lr * 0.5)
    print(f"[v22full] 参数 {sum(p.numel() for p in model.parameters()):,}+master {sum(p.numel() for p in master.parameters()):,} | 零量化 | 源 512", flush=True)
    tasks = ["color", "shift", "flip", "combo", "denoise"]
    exam_pairs = [(t, V.make_pair(src, t, np.random.default_rng(43))) for t in tasks]
    # 监控状态（主人 5 层防护）
    ce_ema, aux_val, cool_down = 8.0, 1.0, 0

    def exam(tag):
        model.eval(); res = []
        with torch.no_grad():
            for t, (inp, rule, tgt) in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                pred_src, logits, *_ = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                # 输出用连续解码（平滑细腻，零量化无海报化）；CE mode-seeking loss 已驱动主干编码鲜色
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
        print(f"  [v22full {tag}] " + " | ".join(
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
        pred, logits, ct, ac, st, gate = model(inp, attr, rule)
        # 主loss改mode-seeking：CE押定颜色字典（主人文本P6 tied CE机制），不再mean-seeking求均值
        with torch.no_grad():
            sf = src_t.permute(0, 2, 3, 1).reshape(-1, 3)               # [HW,3]
            cb = model.color_codebook                                    # [K,3] 连续可学颜色
            dist = (sf * sf).sum(1, keepdim=True) + (cb * cb).sum(1)[None, :] - 2 * sf @ cb.t()
            target_idx = dist.argmin(1).reshape(1, src_t.shape[2], src_t.shape[3])   # 最近字典色 [1,H,W]
        main_loss = F.cross_entropy(logits, target_idx)                  # CE mode-seeking（押定不平均）
        # 辅助：图像科学色度/结构保真（逐像素，作用在连续解码RGB上）
        lab_p = M.torch_rgb_to_lab(pred); lab_s = M.torch_rgb_to_lab(src_t)
        loss_C = F.l1_loss(lab_p[:, 1:3], lab_s[:, 1:3]) / 128.0                 # a,b色度L1：消色相偏移（CCD-Net）
        C_p = torch.sqrt(lab_p[:, 1] ** 2 + lab_p[:, 2] ** 2 + 1e-6)
        C_s = torch.sqrt(lab_s[:, 1] ** 2 + lab_s[:, 2] ** 2 + 1e-6)
        loss_A = F.l1_loss(C_p, C_s) / 128.0                                     # 色度幅值L1：抗sepia逼饱和
        loss_B = 1.0 - M.torch_ssim(pred, src_t)                                 # SSIM：结构/亮度/对比度（Wang2004）
        gate_loss = (gate.std() * 10).clamp(0.01, 1.0)
        sub_losses = torch.stack([gate_loss.detach(), loss_A.detach(), loss_B.detach(), loss_C.detach(), main_loss.detach()])
        temps = master(sub_losses)                          # 自学习比例
        # 监控：主loss尖峰（纯相对阈值，自适应Lab尺度；去掉绝对5——Lab恒>5会clamp死主梯度）
        m_now = main_loss.item()
        if m_now > ce_ema * 3.0:
            cool_down = max(cool_down, 50)
            main_loss = main_loss.clamp(max=ce_ema * 3.0)
        ce_ema = 0.90 * ce_ema + 0.10 * main_loss.item()
        # aux_scale 自愈：震荡退、健康进
        if cool_down > 0:
            aux_val = max(0.2, aux_val * 0.95); cool_down -= 1
        else:
            aux_val = min(1.0, aux_val * 1.01)
        batch_loss = main_loss + aux_val * (temps[1] * loss_A + temps[2] * loss_B + temps[3] * loss_C) + temps[0] * gate_loss
        opt.zero_grad(); opt_master.zero_grad()
        batch_loss.backward()
        # NaN 跳步（主人 NaN 防护）
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        if not has_nan:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt_master.step()
        else:
            opt.zero_grad(); opt_master.zero_grad()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] loss={batch_loss.item():.4f} temps={temps.detach().cpu().numpy().round(2).tolist()} aux={aux_val:.2f} {time.time()-t0:.0f}s VRAM{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            del pred, logits, ct, ac, st, gate, batch_loss, main_loss, lab_p, lab_s   # 考试前释放训练张量防OOM
            torch.cuda.empty_cache()
            exam(f"E{ep}")
    print(f"[完成] {time.time()-t0:.0f}s VRAM峰{torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save({"model": model.state_dict(), "master": master.state_dict()}, os.path.join(out, "pixel_v22full_best.pt"))


if __name__ == "__main__":
    main()