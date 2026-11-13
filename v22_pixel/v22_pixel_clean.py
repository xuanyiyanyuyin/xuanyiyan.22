# -*- coding: utf-8 -*-
"""V22 像素级图生图 · 干净架构（2026-11-12，按主人最新设计重写，非 M2 补丁）

设计根据（传统图像处理 + 主人 V22.1 球面架构）：
  · 万物皆 Token：像素/属性 → 球面 Embed(F.normalize) + 位置编码 → 同构 token → 一套栈。
  · 像素 = 位置编码 + 色值（RGB 三原色基）。位置编码用光栅/Morton（局部保持）。
  · 颜色规则 = CLUT 颜色组 remap：改调色板条目（颜色组），索引图(位置)不动；
    中性色组(白/灰背景)恒等 ⇒ 背景留白，不是全局乘子滤镜。（索引色/CLUT + CCV）
  · 几何规则 = 位置 gather（平移/翻转/仿射），构造性、可微。
  · P3 图像知识框架：P3Vis（V1色彩/V2几何/V3纹理/V4空间深度/V5语义）。
  · P7 联动 = vMF 球面门控 gate=σ(κ·cos(normalize(h),μ)+b)（主人最新球面件，
    可学偏好方向 μ + 浓度 κ）；加法注入 + 小增益(0.01冷启动→0.1)；融合=拼接+投影。
  · 方向性：文→图(生成)用皮尔逊/线性；匹配用余弦。
  · 结构 loss：像素 L1 + 梯度域 L1(像素排列顺序) + CCV 同色位置组一致性。
  · 网络只做内容重建（退化输入→干净源），规则构造性后施加（规则由构造，网络管内容）。

硬约束：单组课程、不准 OOM、原件只读。本文件是独立干净引擎，不动 M2。
"""
import os, sys, glob, time, math, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from P3_vision.p3vis import P3Vis
import color_science as cs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 64
DATA_DIR = os.path.join(HERE, "img2img_data")
OUT_DIR = os.path.join(HERE, "pixel_clean_out")
os.makedirs(OUT_DIR, exist_ok=True)


def imread_u8(p):
    d = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_COLOR) if d.size else None


def imwrite_u8(p, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(p)


def load_rgb(p):
    im = imread_u8(p)
    if im is None:
        return None
    im = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ════════════════════════ 构造性规则（传统图像） ════════════════════════
# torch Lab 转换（CIE D65，可微，用于保明度 remap）
_M_RGB2XYZ = torch.tensor(cs.M_RGB2XYZ, dtype=torch.float32)
_M_XYZ2RGB = torch.tensor(cs.M_XYZ2RGB, dtype=torch.float32)
_D65 = torch.tensor(cs.D65, dtype=torch.float32)


def torch_srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)


def torch_linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0) ** (1 / 2.4) - 0.055)


def torch_rgb_to_lab(rgb):
    lin = torch_srgb_to_linear(rgb)
    xyz = torch.einsum('ij,bjhw->bihw', _M_RGB2XYZ.to(rgb.device), lin)
    xyz_w = xyz / _D65.to(rgb.device).view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz_w > d ** 3, xyz_w.clamp(min=1e-8) ** (1 / 3), xyz_w / (3 * d ** 2) + 4 / 29)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def torch_lab_to_rgb(lab):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    d = 6 / 29
    finv = lambda t: torch.where(t > d, t ** 3, 3 * d ** 2 * (t - 4 / 29))
    xyz = torch.stack([finv(fx), finv(fy), finv(fz)], 1) * _D65.to(lab.device).view(1, 3, 1, 1)
    lin = torch.einsum('ij,bjhw->bihw', _M_XYZ2RGB.to(lab.device), xyz)
    return torch_linear_to_srgb(lin.clamp(0, 1))


# 颜色组 remap（CLUT 语义）：中性色组(背景)恒等，只 remap 彩色组。
# ★Lab 保明度：保 L*（CIE 明度，比 HSV-V 更感知均匀），只改 a*b* 色度；饱和度权重 w 保护中性组。
def color_group_remap(rgb, mult):
    """rgb [B,3,H,W] ∈[0,1], mult [3]。Lab 空间保 L* 改色度，中性组(背景)恒等。"""
    mult = torch.as_tensor(mult, dtype=rgb.dtype, device=rgb.device).reshape(1, 3, 1, 1)
    sat = rgb.max(1, keepdim=True).values - rgb.min(1, keepdim=True).values
    w = (sat * 4.0).clamp(0.0, 1.0)                      # 组权重：中性→0，彩色→1
    lab_o = torch_rgb_to_lab(rgb)
    multiplied = (rgb * (1.0 + w * (mult - 1.0))).clamp(0.0, 1.0)
    lab_m = torch_rgb_to_lab(multiplied)
    L = lab_o[:, 0:1]                                    # 保明度 L*
    a = lab_o[:, 1:2] + w * (lab_m[:, 1:2] - lab_o[:, 1:2])
    b = lab_o[:, 2:3] + w * (lab_m[:, 2:3] - lab_o[:, 2:3])
    return torch_lab_to_rgb(torch.cat([L, a, b], 1)).clamp(0.0, 1.0)


def geo_gather(rgb, kind, param=None):
    """几何位置 gather（构造性、可微）。rgb [B,3,H,W]。kind: shift/flip/rot90/affine。"""
    if kind == "shift":
        dy, dx = param
        return torch.roll(rgb, shifts=(dy, dx), dims=(2, 3))
    if kind == "flip":
        return torch.flip(rgb, dims=[3])
    if kind == "vflip":
        return torch.flip(rgb, dims=[2])
    if kind == "rot90":
        return torch.rot90(rgb, k=1, dims=(2, 3))
    if kind == "affine":
        a, b, c, d, e, f = param
        B, _, H, W = rgb.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=rgb.device, dtype=rgb.dtype),
                                torch.arange(W, device=rgb.device, dtype=rgb.dtype), indexing="ij")
        sx = (a * xx + b * yy + e).clamp(0, W - 1)
        sy = (c * xx + d * yy + f).clamp(0, H - 1)
        gx = sx / (W - 1) * 2 - 1
        gy = sy / (H - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return rgb


# 颜色名 → 乘子（CLUT 调色板条目偏移）。背景组恒等由 color_group_remap 保证。
COLOR_MULT = {"偏红": (1.0, 0.55, 0.55), "偏绿": (0.55, 1.0, 0.55), "变暗": (0.7, 0.7, 0.7),
              "暖调": (1.0, 0.85, 0.70), "冷调": (0.80, 0.90, 1.00)}


# ════════════════════════ 损失（纯线性光 L1，零量化零平滑） ════════════════════════
def srgb_to_linear(x):
    """sRGB 编码 → 线性光（IEC 61966-2-1）。物理正确的光强空间。"""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)


def _gauss_win(win=11, sigma=1.5, channels=3, device=None):
    x = torch.arange(win, dtype=torch.float32, device=device) - win // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    return k2.expand(channels, 1, win, win).contiguous()


def torch_ssim(pred, tgt, win=11):
    """可微 SSIM（data_range=1），管亮度/对比度/结构。pred,tgt [B,3,H,W]。"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss_win(win, 1.5, pred.shape[1], pred.device)
    pad = win // 2
    mu_p = F.conv2d(pred, k, padding=pad, groups=pred.shape[1])
    mu_t = F.conv2d(tgt, k, padding=pad, groups=pred.shape[1])
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sp = F.conv2d(pred * pred, k, padding=pad, groups=pred.shape[1]) - mu_p2
    st = F.conv2d(tgt * tgt, k, padding=pad, groups=pred.shape[1]) - mu_t2
    spt = F.conv2d(pred * tgt, k, padding=pad, groups=pred.shape[1]) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * spt + C2)) / ((mu_p2 + mu_t2 + C1) * (sp + st + C2))
    return ssim_map.mean()


def structural_loss(pred, tgt, no_smooth=True, chroma_weight=0.5, hl_weight=1.0, sh_weight=1.0, ssim_weight=0.0):
    """pred,tgt [B,3,H,W] sRGB。线性光 L1 + ② 色度押注项（Lab a/b，恢复 CE 的 mode-seeking，惩罚发灰）。
       零量化、零平滑（主人令）。clamp 激活去 sigmoid 天花板。a/b 除 128 归一到 CIE 单位量纲。"""
    pl = srgb_to_linear(pred); tl = srgb_to_linear(tgt)
    w = torch.ones_like(tl)
    w = torch.where(tl > 0.6, hl_weight, w)
    w = torch.where(tl < 0.05, sh_weight, w)
    loss = (torch.abs(pl - tl) * w).mean()
    if chroma_weight > 0:
        lab_p = torch_rgb_to_lab(pred); lab_t = torch_rgb_to_lab(tgt)
        loss = loss + chroma_weight * F.l1_loss(lab_p[:, 1:3] / 128.0, lab_t[:, 1:3] / 128.0)
    if ssim_weight > 0:
        loss = loss + ssim_weight * (1.0 - torch_ssim(pred, tgt))
    return loss


# ════════════════════════ vMF 球面门控（主人最新球面件） ════════════════════════
class vMFGate(nn.Module):
    """gate_i = σ(κ_i·cos(normalize(h), μ_i) + b_i)。可学偏好方向 μ + 浓度 κ。
       来源 v22_vmf_modules.py P7Gate_vMF/TurboMeta_vMF（最新球面归一化）。"""
    def __init__(self, in_dim, out_dim, hidden=96):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.mu = nn.Parameter(torch.randn(out_dim) * 0.1)
        self.kappa = nn.Parameter(torch.ones(out_dim) * 2.0)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        h = self.fc(x)
        h_norm = F.normalize(h, dim=-1, eps=1e-8)
        mu_norm = F.normalize(self.mu, dim=0, eps=1e-8)
        cos = h_norm * mu_norm.unsqueeze(0)
        return torch.sigmoid(self.kappa * cos + self.bias)


# ════════════════════════ 干净像素引擎 ════════════════════════
class PixelEngine(nn.Module):
    """像素 CNN 基底 + P3 知识 + vMF 球面联动 + 连续 RGB 解码。
       网络重建干净源；规则（颜色组 remap + 几何 gather）构造性后施加。"""
    def __init__(self, dim=64, attr_dim=256, rule_dim=16):
        super().__init__()
        self.p3vis = P3Vis(chroma=True)                    # ③ 色度分布特征开（上色监督在色度分布，p3vis:327）
        # 像素编码器：CNN（局部像素联动），GroupNorm/GELU（现代化，非到处球面）
        self.enc = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        # P3 属性 → 球面 Embed（万物皆 Token：注入表征球面单位化）
        self.attr_proj = nn.Linear(attr_dim, dim)
        # 规则 spec → 球面 Embed
        self.rule_proj = nn.Linear(rule_dim, dim)
        # P7 联动：vMF 球面门控，条件 = 拼接(属性, 规则)（融合=拼接+投影）
        self.p7_gate = vMFGate(dim * 2, dim)
        # 加法注入增益（冷启动 0.01 → 训练后 0.1，主人规格）
        self.inject_gain = nn.Parameter(torch.tensor(0.01))
        # ① 逐位身份门控（port 原件 v4, train_v22_img2img:235）+ ③ 球面归一化+可学习尺度（v4.1, :234）
        self.id_proj = nn.Linear(dim * 2, dim)             # 全局身份（attr+rule 融合）
        self.h_scale = nn.Parameter(torch.tensor(2.0))     # 球面归一化后的可学习尺度
        self.pos_gate = nn.Conv2d(dim, dim, 1)             # 逐位身份门 tanh(clamp(·,-2,2))
        nn.init.xavier_uniform_(self.pos_gate.weight, gain=0.1)
        self.dim = dim
        # 像素解码器：连续 RGB
        self.dec = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, 3, 3, padding=1))

    def encode_attr(self, img_bgr_list):
        attrs = []
        for im in img_bgr_list:
            a = self.p3vis.process_image(im)
            attrs.append(torch.tensor(a, dtype=torch.float32))
        return torch.stack(attrs).to(DEV)

    def _pos_enc(self, H, W, device):
        # 正弦坐标编码 [1,dim,H,W]，分辨率无关（替代原件固定 n_pos 查表，CNN 可变分辨率）
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        n = max(self.dim // 4, 1)
        freqs = torch.exp(torch.arange(n, device=device) * (-math.log(10000.0) / max(n - 1, 1)))
        feats = []
        for f in freqs:
            feats += [torch.sin(gx * f * math.pi), torch.cos(gx * f * math.pi),
                      torch.sin(gy * f * math.pi), torch.cos(gy * f * math.pi)]
        return torch.stack(feats, 0)[:self.dim].unsqueeze(0)

    def forward(self, x, attr, rule):
        # x [B,3,H,W] 退化输入；attr [B,256] P3知识；rule [B,16] 规则spec
        B, _, H, W = x.shape
        h = self.enc(x)                                   # [B,dim,H,W] 像素特征
        a_emb = F.normalize(self.attr_proj(attr), dim=-1)  # 球面属性 token
        r_emb = F.normalize(self.rule_proj(rule), dim=-1)  # 球面规则 token
        cond = torch.cat([a_emb, r_emb], dim=-1)          # 融合=拼接
        gate = self.p7_gate(cond)                         # [B,dim] 全局 vMF 球面联动（P7）
        # ③ 全局身份：球面归一化 + 可学习尺度（原件 v4.1）
        h_id = self.h_scale * F.normalize(self.id_proj(cond), dim=-1)   # [B,dim]
        # ① 逐位身份门控（原件 v4）：门按位置调制 ⇒ 逐位调色记忆
        pos = self._pos_enc(H, W, x.device)               # [1,dim,H,W]
        g_pos = torch.tanh(torch.clamp(self.pos_gate(pos), -2.0, 2.0))  # [B,dim,H,W]
        hp = h_id[:, :, None, None] * g_pos               # [B,dim,H,W] 逐位身份注入
        # 所有注入套 inject_gain 冷启动（0.01→0.1），防满强度扰动把 dec 推越界
        h = h + self.inject_gain * (gate[:, :, None, None] * h + hp + 0.1 * pos)
        out = self.dec(h)
        # 直通 clamp：前向削到[0,1]出纯白/纯黑，反向恒等梯度 ⇒ 无死区、不塌（修连续 clamp 梯度死区崩溃）
        return out + (out.clamp(0.0, 1.0) - out).detach()


# ════════════════════════ 数据（单组课程） ════════════════════════
def build_dataset():
    files = {os.path.basename(f).rsplit(".", 1)[0]: f for f in glob.glob(os.path.join(DATA_DIR, "*.bmp"))}
    train_src = load_rgb(files["apple_01"])              # 单组：只训 apple_01
    exam_names = ["apple_01", "apple_16", "apple_17", "apple_18", "apple_19", "apple_20", "sun_05"]
    exam_src = {n: load_rgb(files[n]) for n in exam_names if n in files}
    return train_src, exam_src


def make_pair(src, task, rng):
    """生成 (退化输入, 干净源, 规则spec, 目标)。规则构造性施加于干净源得目标。"""
    H = W = IMG
    src_t = torch.tensor(src, dtype=torch.float32).permute(2, 0, 1)   # [3,H,W]
    rule = np.zeros(16, dtype=np.float32)
    if task == "color":
        name = rng.choice(list(COLOR_MULT))
        mult = COLOR_MULT[name]
        rule[0] = 1.0
        rule[1:4] = mult
        gray = (src @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        inp = np.stack([gray, gray, gray], -1)
        tgt = color_group_remap(src_t.unsqueeze(0), mult)[0]
    elif task == "shift":
        dy, dx = int(rng.integers(2, 15)), int(rng.integers(2, 15))
        rule[4] = 1.0; rule[5] = dy / IMG; rule[6] = dx / IMG
        inp = src.copy()
        tgt = geo_gather(src_t.unsqueeze(0), "shift", (dy, dx))[0]
    elif task == "flip":
        rule[7] = 1.0
        inp = src.copy()
        tgt = geo_gather(src_t.unsqueeze(0), "flip")[0]
    elif task == "combo":
        name = rng.choice(["偏红", "暖调", "冷调"])
        mult = COLOR_MULT[name]
        dy, dx = int(rng.integers(2, 15)), int(rng.integers(2, 15))
        rule[0] = 1.0; rule[1:4] = mult; rule[4] = 1.0; rule[5] = dy / IMG; rule[6] = dx / IMG
        gray = (src @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        inp = np.stack([gray, gray, gray], -1)
        tgt = color_group_remap(src_t.unsqueeze(0), mult)
        tgt = geo_gather(tgt, "shift", (dy, dx))[0]
    else:  # denoise
        sigma = float(rng.choice([0.1, 0.2, 0.3]))
        rule[8] = 1.0; rule[9] = sigma
        inp = np.clip(src + rng.normal(0, sigma, src.shape).astype(np.float32), 0, 1)
        tgt = src_t
    inp_t = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1)
    return inp_t, src_t, torch.tensor(rule), tgt


def apply_rule_to_pred(pred, rule):
    """把构造性规则施加到网络重建的干净源 pred [B,3,H,W]，得变换预测。与目标同口径。
       逐样本收集+stack，无原地操作（保 autograd）。"""
    results = []
    for i in range(pred.shape[0]):
        o = pred[i]
        r = rule[i]
        if r[0] > 0.5:
            o = color_group_remap(o.unsqueeze(0), r[1:4])[0]
        if r[4] > 0.5:
            dy, dx = int(round(float(r[5]) * IMG)), int(round(float(r[6]) * IMG))
            o = geo_gather(o.unsqueeze(0), "shift", (dy, dx))[0]
        if r[7] > 0.5:
            o = geo_gather(o.unsqueeze(0), "flip")[0]
        results.append(o)
    return torch.stack(results)


# ════════════════════════ 训练 + 评测 ════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--exam_every", type=int, default=20)
    ap.add_argument("--no_smooth", type=int, default=1)   # 主人令：不平滑。平滑项(梯度+CCV)是暗化主因，纯L1最佳
    ap.add_argument("--render", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6.0 / (torch.cuda.get_device_properties(0).total_memory / 2**30)))

    train_src, exam_src = build_dataset()
    model = PixelEngine(dim=args.dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数 {n_params:,} | 设备 {DEV} | 单组训练源 apple_01", flush=True)

    tasks = ["color", "shift", "flip", "combo", "denoise"]
    exam_pairs = []
    for nm, src in exam_src.items():
        for t in tasks:
            exam_pairs.append((nm, t, make_pair(src, t, np.random.default_rng(args.seed + 1))))

    def exam(tag):
        model.eval()
        rows = []
        with torch.no_grad():
            for nm, t, (inp, src, rule, tgt) in exam_pairs:
                attr = model.encode_attr([(inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]])
                pred_src = model(inp.unsqueeze(0).to(DEV), attr, rule.unsqueeze(0).to(DEV))
                pred = apply_rule_to_pred(pred_src, rule.unsqueeze(0).to(DEV))[0]
                tg = tgt.to(DEV)
                l1 = float((pred - tg).abs().mean())
                pc = pred.reshape(-1).cpu().numpy(); tc = tg.reshape(-1).cpu().numpy()
                pear = float(np.corrcoef(pc, tc)[0, 1]) if pc.std() > 1e-8 else float("nan")
                p_np = pred.permute(1, 2, 0).cpu().numpy(); t_np = tg.permute(1, 2, 0).cpu().numpy()
                de00 = float(cs.delta_e_2000(cs.rgb_to_lab(p_np), cs.rgb_to_lab(t_np)).mean())
                ss = cs.ssim(p_np, t_np)
                rows.append((nm, t, l1, pear, de00, ss, pred, tg, inp))
        model.train()
        by_task = {}
        for nm, t, l1, pear, de00, ss, *_ in rows:
            by_task.setdefault(t, []).append((l1, pear, de00, ss))
        all_l1 = np.mean([r[2] for r in rows]); all_p = np.nanmean([r[3] for r in rows])
        all_de = np.mean([r[4] for r in rows]); all_ss = np.mean([r[5] for r in rows])
        print(f"  [i2i {tag}] L1={all_l1:.4f} 皮尔逊={all_p:.4f} ΔE00={all_de:.2f} SSIM={all_ss:.4f}", flush=True)
        for t, vs in by_task.items():
            print(f"    {t:8s} L1={np.mean([v[0] for v in vs]):.4f} 皮尔逊={np.nanmean([v[1] for v in vs]):.4f} "
                  f"ΔE00={np.mean([v[2] for v in vs]):.2f} SSIM={np.mean([v[3] for v in vs]):.4f}", flush=True)
        if args.render and tag.startswith("E"):
            for nm, t, l1, pear, de00, ss, pred, tg, inp in rows:
                if nm != "apple_01":
                    continue
                trip = torch.cat([inp, pred.cpu(), tg.cpu()], dim=2).permute(1, 2, 0).numpy()
                trip_d = cs.floyd_steinberg(np.clip(trip, 0, 1))   # 输出端抖动消 banding
                big = cv2.resize((trip_d * 255).round().astype(np.uint8), (IMG * 3 * 8, IMG * 8), interpolation=cv2.INTER_NEAREST)
                cs.save_png_with_icc(os.path.join(OUT_DIR, f"{tag}_{nm}_{t}.png"), big)   # big 是 RGB，PIL 要 RGB（勿转 BGR）
        return all_l1

    print("[考试] 训练前基线", flush=True)
    exam("base")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        batch = [make_pair(train_src, tasks[i % len(tasks)], rng) for i in range(args.bs)]
        inp = torch.stack([b[0] for b in batch]).to(DEV)
        src = torch.stack([b[1] for b in batch]).to(DEV)
        rule = torch.stack([b[2] for b in batch]).to(DEV)
        tgt = torch.stack([b[3] for b in batch]).to(DEV)
        attr = model.encode_attr([(b[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1] for b in batch])
        pred_src = model(inp, attr, rule)
        pred = apply_rule_to_pred(pred_src, rule)
        loss = structural_loss(pred, tgt, no_smooth=args.no_smooth)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % args.exam_every == 0 or ep == args.epochs:
            print(f"[E{ep}] loss={loss.item():.4f} inject_gain={model.inject_gain.item():.3f} "
                  f"kappa_mean={model.p7_gate.kappa.mean().item():.2f} {time.time()-t0:.0f}s", flush=True)
            exam(f"E{ep}")
    print(f"[完成] 总耗时 {time.time()-t0:.0f}s | GPU峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "pixel_clean_best.pt"))


if __name__ == "__main__":
    main()