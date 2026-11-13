"""V22.C1 — 连续域（像素级、零量化）图生图引擎。新副本；`_reference_m2_copy.py` 是 1cd2fd84 的原样参考。

架构逐块保留：P7 跨 patch 上下文、P3Vis 属性、球面归一化身份 + 有界增益条件注入、
无缝邻域 gather + 自学习路由、patch 内多头注意力、copy/transform 门控、逐任务门偏置。

表示层整体换成连续域：
  输入   to_bins→int→bin_embed 查表    →  逐位标量 Fourier 编码（无词表、无 bin）
  变形   16×16 档轴置换矩阵连乘         →  逐通道解析逆 x/exp(Σ_k sel_k·a_kc)，组合精确
  输出   16 类 logits + argmax          →  逐位连续标量 sigmoid ∈[0,1]
  损失   CE                             →  L1（色度/亮度分项独立记账，不受开关门控）
  打分   pixel_acc                      →  像素 L1% + 逐图中位皮尔逊；旧 bin-acc 仅在末端算一次供对齐

仪器闸 8 条，全部 assert / sys.exit，不写成注释。其中闸6 按手册教训实现：
**恒等初始化的通路必须先把参数扰动离初值，再做 on/off 差分，用完逐位还原**——
否则"关掉它输出不变"是初值性质，不是假连接。
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
import sys, glob, time, random, argparse
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE, BASE_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3_vision"))
from p3vis import P3Vis
from P7_cross_sent.model import P7WordRouter2048

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--lr", type=float, default=0.012)
ap.add_argument("--clip_norm", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--k_labeled", type=int, default=8)
ap.add_argument("--use_generic", type=int, default=1)
ap.add_argument("--gen_max", type=int, default=300)
ap.add_argument("--exam_n_generic", type=int, default=30)
ap.add_argument("--exam_every", type=int, default=50)
ap.add_argument("--max_sources", type=int, default=0)
ap.add_argument("--smoke", type=int, default=0)
ap.add_argument("--gates_only", type=int, default=0)
ap.add_argument("--exam_only_labeled", type=int, default=0)  # [C3同卷] 1=考卷只留那 7 张带标签图(与旧量化版 65.15 同一批图)，训练池一字不动
ap.add_argument("--embed_dim", type=int, default=256)
ap.add_argument("--encoder_width", type=int, default=512)
ap.add_argument("--fourier_k", type=int, default=8)
ap.add_argument("--s_seamless", type=int, default=4)
ap.add_argument("--sn_lin_n", type=int, default=0)  # [s8几何/容量分离] 0=每偏移各一个线性头(现行为)；>0=偏移照旧但只建这么多头、按下标取模共享
ap.add_argument("--attn_heads", type=int, default=4)
ap.add_argument("--cont_head", type=str, default="rgb", choices=["rgb", "ycb"])
ap.add_argument("--prompt_cond", type=int, default=1)
ap.add_argument("--task_cond", type=int, default=1)
ap.add_argument("--gate_task", type=int, default=1)
ap.add_argument("--h_scale", type=float, default=1.0)
ap.add_argument("--n_bins_legacy", type=int, default=16)
ap.add_argument("--brake_after", type=int, default=50)   # 过了这么多步还不赢恒等地板 ⇒ 响亮停机，不许把地板头当成绩
ap.add_argument("--detect_anomaly", type=int, default=0)  # 只在查 NaN 来源时开：逐算子回溯，慢，会点名是哪个 op  produce 了非有限梯度
ap.add_argument("--save_dir", type=str, default="img2img_cont")
ap.add_argument("--engine_dir", type=str, default=r"D:/ai/lingbot_phase2_backup/v22_stage1")
ap.add_argument("--generic_dir", type=str, default=r"D:/ai/output/images")
args = ap.parse_args()

SEED = args.seed
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG, PATCH, NPATCH = 64, 16, 16
N_IN = PATCH * PATCH * 3
assert NPATCH * N_IN == IMG * IMG * 3

LOG_DIR = os.path.join(BASE_DIR, args.save_dir)
os.makedirs(LOG_DIR, exist_ok=True)
_log = open(os.path.join(LOG_DIR, "run_%d.txt" % SEED), "a", encoding="utf-8")
T0 = time.time()


def w(s):
    _log.write(s + "\n"); _log.flush(); print(s, flush=True)


PROBE = {"on": 0}
GB = 2 ** 30
GATE_MEAN = [float("nan")]   # copy 门平均开度，只在评测（no_grad）时取样
GATE_ROWS = []               # 评测期逐像素门开度，按任务分开报（总分掩盖遮补就是这么来的）
HEAD_RAW = [float("nan")]    # 残差头进 tanh 前的平均幅度
HEAD_SAT = [float("nan")]    # 残差头饱和（|raw|>2）像素占比
Q_STD = [float("nan")]       # 进头之前的特征 std，用来判『门/头消费的是归一化 q 还是裸 q』
PART_STD = [(float("nan"),) * 4]  # q 里四项的尺度：val / attr_proj / pctx_proj / h
Q_MIN = [None]                   # ‖q‖ 的最小值（存张量不 .item()，取用时才同步）
EXAM_PREV = [None]           # 上一次考试的原样预测，用来量『读数到底动不动』


def mem(tag):
    """显存分段探针：只在被点名的那一步打印，量『谁把卡吃满』，不靠猜。"""
    if not PROBE["on"] or not str(DEVICE).startswith("cuda"):
        return
    w("    [显存] %-12s 在活=%6.2fGiB 本步峰值=%6.2fGiB 已预留=%6.2fGiB" % (
        tag, torch.cuda.memory_allocated() / GB, torch.cuda.max_memory_allocated() / GB,
        torch.cuda.memory_reserved() / GB))


IN_MODEL_PATH = [0]


def imread_u8(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_rgb(p):
    img = imread_u8(p)
    if img is None:
        return None
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, img.copy()


def to_patch_major(rgb):
    return rgb.reshape(4, 16, 4, 16, 3).transpose(0, 2, 1, 3, 4).reshape(NPATCH, N_IN)


def from_patch_major(flat):
    return np.asarray(flat).reshape(4, 4, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(IMG, IMG, 3)


def to_bins_legacy(x, nb=None):
    """只允许出现在打分末端。模型路径里被调用 = 表示层没换干净。"""
    if IN_MODEL_PATH[0]:
        raise AssertionError("旧量化器出现在模型路径里")
    nb = int(nb or args.n_bins_legacy)
    return np.clip((np.asarray(x, np.float32) * nb).astype(np.int64), 0, nb - 1)


TASKS = ["去噪", "遮补", "上色"]
SIGMA_POOL = np.array([0.10, 0.20, 0.30], np.float32)
T_NAMES = ["偏红", "偏绿", "变暗", "翻转", "平移"]
NT = len(T_NAMES)
MULT_TABLE = np.array([[1.0, .55, .55], [.55, 1.0, .55], [.7, .7, .7], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], np.float32)
EXAM_COMBOS = [(0, 3), (1, 4), (2, 3), (0, 2), (1, 3), (0, 4), (2, 4)]


def mhot(names):
    v = np.zeros(NT, np.float32)
    for k in names:
        v[int(k)] = 1.0
    return v


def apply_combo(rgb, names):
    out = rgb.copy()
    m = np.ones(3, np.float32)
    for k in names:
        m *= MULT_TABLE[int(k)]
    if not np.allclose(m, 1.0):
        out = np.clip(out * m, 0, 1).astype(np.float32)
    if any(int(k) == 3 for k in names):
        out = np.fliplr(out).copy()
    if any(int(k) == 4 for k in names):
        out = np.roll(out, (8, 8), axis=(0, 1))
    return out.astype(np.float32)


def degrade(rgb, task, rng):
    out, sigma = rgb.copy(), 0.0
    if task == 0:
        sigma = float(rng.choice(SIGMA_POOL))
        out = np.clip(rgb + rng.normal(0, sigma, rgb.shape).astype(np.float32), 0, 1)
    elif task == 1:
        k = int(rng.choice([16, 24, 32]))
        y, x = int(rng.integers(0, IMG - k + 1)), int(rng.integers(0, IMG - k + 1))
        out[y:y + k, x:x + k] = 0.5
    else:
        g = (rgb @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.float32)
        out = np.stack([g, g, g], -1)
    return np.clip(out, 0, 1).astype(np.float32), sigma


LABELED_DIR = os.path.join(args.engine_dir, "img2img_data")
GENERIC_DIR = os.path.join(args.engine_dir, "output", "images")
if not os.path.isdir(GENERIC_DIR) and os.path.isdir(args.generic_dir):
    GENERIC_DIR = args.generic_dir
w(f"[数据目录] 带标签={LABELED_DIR} 通用池={GENERIC_DIR}")
labeled_files = sorted(sum([glob.glob(os.path.join(LABELED_DIR, e)) for e in ("*.bmp", "*.png", "*.jpg")], []))
assert labeled_files, "带标签数据没找到: %s" % LABELED_DIR
if args.max_sources:
    labeled_files = labeled_files[:args.max_sources]
gf = sorted(sum([glob.glob(os.path.join(GENERIC_DIR, "**", e), recursive=True)
                 for e in ("*.jpg", "*.png", "*.jpeg")], [])) if args.use_generic else []
if gf:
    rg = np.random.default_rng(SEED)
    gf = list(np.array(gf)[rg.permutation(len(gf))][:args.gen_max + args.exam_n_generic])

sources, generic_pool, bad = [], [], 0
for f in labeled_files:
    r = load_rgb(f)
    if r is None:
        bad += 1; continue
    sources.append({"name": os.path.basename(f), "rgb": r[0], "bgr": r[1], "rgb_pm": to_patch_major(r[0]).reshape(-1)})
for f in gf:
    r = load_rgb(f)
    if r is None:
        bad += 1; continue
    generic_pool.append({"name": os.path.basename(f), "rgb": r[0], "bgr": r[1], "rgb_pm": to_patch_major(r[0]).reshape(-1)})

EXAM_NAMES = {f"apple_{i:02d}.bmp" for i in range(16, 21)} | {"sun_05.bmp", "apple_01.bmp"}
exam_labeled = [s for s in sources if s["name"] in EXAM_NAMES]
train_labeled = [s for s in sources if s["name"] not in EXAM_NAMES or s["name"] == "apple_01.bmp"]
exam_generic = generic_pool[-args.exam_n_generic:] if args.use_generic else []
train_generic = generic_pool[:-args.exam_n_generic] if args.use_generic else []
w(f"[数据] 带标签(train/exam)={len(train_labeled)}/{len(exam_labeled)} 通用(train/exam)={len(train_generic)}/{len(exam_generic)} 坏图={bad}")
assert exam_labeled, "考卷图一张都没选上 ⇒ 数据目录或命名变了"

_p3vis = P3Vis(chroma=False, kmeans_seed=SEED)
if _p3vis.tex_backend != "skimage":
    w("[环境不足] 没有 scikit-image ⇒ P3Vis 纹理层整段恒零，停"); sys.exit(3)

_probe = _p3vis.process_image((sources[0]["rgb"] * 255).astype(np.uint8))
ATTR_DIM = int(np.asarray(_probe).size)
w(f"[P3Vis] process_image 输出维度={ATTR_DIM}")


def build_pairs(srcs, k, rng, exam=False):
    D, T, P, A, G, rows = [], [], [], [], [], []
    for s in srcs:
        if exam:
            combos = [(2, tuple(c)) for c in EXAM_COMBOS] + [(t, ()) for t in range(3)]
        else:
            combos = [(int(rng.integers(0, 3)),
                       tuple(rng.choice(NT, size=int(rng.integers(0, 3)), replace=False).tolist()))
                      for _ in range(max(1, k))]
        for task, names in combos:
            base = apply_combo(s["rgb"], names)
            deg, sigma = degrade(base, task, rng)
            D.append(to_patch_major(deg).reshape(-1)); T.append(s["rgb_pm"]); P.append(mhot(names))
            G.append(0 if sigma == 0 else int(np.argmin(np.abs(SIGMA_POOL - sigma))) + 1)
            A.append(np.nan_to_num(np.asarray(_p3vis.process_image((deg * 255).astype(np.uint8)),
                                              np.float32).reshape(-1), nan=0.0))
            rows.append((s["name"], int(task), tuple(int(x) for x in names)))
    return dict(deg=np.stack(D).astype(np.float32), tgt=np.stack(T).astype(np.float32),
                prompt=np.stack(P).astype(np.float32), attr=np.stack(A).astype(np.float32),
                sig=np.array(G, np.int64), rows=rows)


class ValEnc(nn.Module):
    def __init__(self, ed, K):
        super().__init__()
        self.register_buffer("freqs", torch.tensor([2.0 ** i for i in range(K)], dtype=torch.float32) * float(np.pi))
        self.proj = nn.Linear(2 * K + 1, ed, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.radius = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = x.unsqueeze(-1) * self.freqs
        return self.radius * F.normalize(self.proj(torch.cat([x.unsqueeze(-1), torch.sin(z), torch.cos(z)], -1)), dim=-1)


def ycb_to_rgb(ycb):
    y, cb, cr = ycb[..., 0], ycb[..., 1], ycb[..., 2]
    return torch.stack([y + 1.402 * (cr - .5),
                        y - .344136 * (cb - .5) - .7141362862 * (cr - .5),
                        y + 1.772 * (cb - .5)], -1)


def rgb_to_ycb(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return torch.stack([.299 * r + .587 * g + .114 * b,
                        -.168736 * r - .331264 * g + .5 * b + .5,
                        .5 * r - .418688 * g - .081312 * b + .5], -1)


class I2ICont(nn.Module):
    def __init__(self):
        super().__init__()
        ed = self.ed = args.embed_dim
        self.val = ValEnc(ed, args.fourier_k)
        self.attr_proj = nn.Linear(ATTR_DIM, ed, bias=False); nn.init.xavier_uniform_(self.attr_proj.weight, .1)
        self.pctx_proj = nn.Linear(ed, ed, bias=False); nn.init.xavier_uniform_(self.pctx_proj.weight, .1)
        self.input_proj = nn.Linear(256 + ATTR_DIM, ed, bias=False)
        self.encoder = nn.Sequential(nn.Linear(ed, args.encoder_width), nn.GELU(),
                                     nn.Linear(args.encoder_width, ed), nn.GELU())
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .5); nn.init.zeros_(m.bias)
        self.patch_pos = nn.Parameter(torch.randn(NPATCH, ed) * .1)
        self.inpatch_pos = nn.Parameter(torch.randn(N_IN, ed) * .1)
        # 注入侧尺度失控实测：pctx_proj 的 std 从 74（E3）涨到 2364（E50），val 始终 0.059
        # ⇒ 归一化后的方向里像素值占 0.06/2364，读出头基本看不见逐像素信息，恰好是
        # 「上色会、去噪遮补输」的机制。照他引擎『球面归一化身份＋有界增益注入』句式，
        # 三个输入项各自 normalize 后乘 2*sigmoid(gain)，增益初值 0 ⇒ 各项同尺度起步。
        self.gain_val = nn.Parameter(torch.zeros(1))
        self.gain_attr = nn.Parameter(torch.zeros(1))
        self.gain_pctx = nn.Parameter(torch.zeros(1))
        self.h_scale = nn.Parameter(torch.tensor(float(args.h_scale)))
        self.gate_pair = nn.Linear(ed * 2, ed, bias=False); nn.init.xavier_uniform_(self.gate_pair.weight, .1)
        offs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)][:args.s_seamless]
        self.offs = offs
        idx = torch.zeros(len(offs), NPATCH * N_IN, dtype=torch.long); msk = torch.zeros(len(offs), NPATCH * N_IN)
        for kk, (dy, dx) in enumerate(offs):
            for p in range(NPATCH * N_IN):
                pi, rem = divmod(p, N_IN)
                y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
                c = rem % 3
                py, px = divmod(pi, 4)
                Y, X = py * PATCH + y_in + dy, px * PATCH + x_in + dx
                if 0 <= Y < IMG and 0 <= X < IMG:
                    idx[kk, p] = ((Y // PATCH) * 4 + X // PATCH) * N_IN + (Y % PATCH) * (PATCH * 3) + (X % PATCH) * 3 + c
                    msk[kk, p] = 1.0
        self.register_buffer("sn_idx", idx); self.register_buffer("sn_mask", msk)
        self.sn_lin = nn.ModuleList([nn.Linear(ed, ed, bias=False)
                                      for _ in range(len(offs) if args.sn_lin_n == 0 else args.sn_lin_n)])
        for m in self.sn_lin:
            nn.init.xavier_uniform_(m.weight, .1)
        self.sn_router = nn.Linear(ed, len(offs), bias=False); nn.init.zeros_(self.sn_router.weight)
        self.sn_gain = nn.Parameter(torch.tensor(.1))
        for n in ("attn_q", "attn_k", "attn_v", "attn_o"):
            setattr(self, n, nn.Linear(ed, ed, bias=False)); nn.init.xavier_uniform_(getattr(self, n).weight, .1)
        self.attn_gain = nn.Parameter(torch.tensor(.1))
        self.tgate = nn.Linear(ed, 1, bias=False); nn.init.xavier_uniform_(self.tgate.weight, .1)
        self.tgate_task = nn.Embedding(3, 1); nn.init.zeros_(self.tgate_task.weight)
        if args.task_cond:
            self.task_e = nn.Embedding(3, ed); nn.init.zeros_(self.task_e.weight)
            self.sigma_e = nn.Embedding(4, ed); nn.init.zeros_(self.sigma_e.weight)
        if args.prompt_cond:
            self.prompt_e = nn.Embedding(NT, ed); nn.init.zeros_(self.prompt_e.weight)
            self.cond_gain = nn.Parameter(torch.zeros(1))
        # 连续域变形逆：divisor_c = exp(tanh(cgate)·Σ_k sel_k·logm_kc)，logm 零初始化 ⇒ 恒等。
        # 组合精确：指数相加＝乘子相乘，所以两个名同时请求不需要学交叉项。
        self.logm = nn.Parameter(torch.zeros(NT, 3))
        self.cgate = nn.Parameter(torch.zeros(1))
        # 变换支路改成『退化输入 + 残差』。上一版出的是绝对色：初值下 copy 支路=L1 22.79%、
        # 变换支路=L1 36.68%，门两步就饱和 1.0000，(1-g) 连带把变换支路的梯度一起饿死。
        # 残差头必须**零初始化**：初值下两条支路逐位相等 ⇒ 门没有『一上来就全开』的机械优势，
        # 而 ∂tanh(q·W)/∂W|_{W=0}=q≠0，残差头照样拿满梯度。delta_gain 给有界幅度，不改变恒等起点。
        self.delta_gain = nn.Parameter(torch.tensor(1.0))
        if args.cont_head == "rgb":
            self.head = nn.Linear(ed, 1, bias=False); nn.init.zeros_(self.head.weight)
        else:
            self.head_ycb = nn.Linear(ed, 3, bias=False); nn.init.zeros_(self.head_ycb.weight)

    @staticmethod
    def _gain(p):
        return 2.0 * torch.sigmoid(p)

    def forward(self, deg, sent, attr, pctx, task, sig, prompt, disable=frozenset()):
        IN_MODEL_PATH[0] = 1
        try:
            b = deg.shape[0]
            x = deg.view(b, NPATCH, N_IN)
            vs = self.val(x)
            ass = self.attr_proj(attr).unsqueeze(1).unsqueeze(2)
            tok = self._gain(self.gain_val) * F.normalize(vs, dim=-1) \
                + self._gain(self.gain_attr) * F.normalize(ass, dim=-1) + self.patch_pos.unsqueeze(0).unsqueeze(2)
            raw = self.encoder(self.input_proj(torch.cat([sent, attr], -1)))
            c = torch.zeros_like(raw)
            if args.task_cond and "cond" not in disable:
                c = c + self.task_e(task) + self.sigma_e(sig)
            if args.prompt_cond and "prompt" not in disable:
                c = c + self._gain(self.cond_gain) * (prompt @ self.prompt_e.weight)
            h = self.h_scale * F.normalize(F.normalize(raw, dim=-1) + c, dim=-1)
            pss = self.pctx_proj(pctx).unsqueeze(2)
            tok = tok + h.unsqueeze(1).unsqueeze(2) + self._gain(self.gain_pctx) * F.normalize(pss, dim=-1)
            mem("embed")
            q = tok + self.inpatch_pos.view(1, 1, N_IN, self.ed)
            # 同一件事，但不实体化 [b,NPATCH,N_IN,2ed] 的 pair：把 [q|patch_pos] 的拼接门
            # 写成两段矩阵乘相加（拼接维＝权重列的分块），patch_pos 那段只有 [NPATCH,ed]。
            Wg = self.gate_pair.weight
            g = q @ Wg[:, :self.ed].t() + (self.patch_pos @ Wg[:, self.ed:].t()).view(1, NPATCH, 1, self.ed)
            q = q * torch.sigmoid(g)
            # 球面归一化身份＋有界增益：token 流每个注入点之后都归一化。
            # 裸 q 实测 std=4940（q 里四项 val=0.064 attr_proj=4.49 pctx_proj=8.06 h=0.056，尺度病在注入侧不在像素侧），
            # 后果两条：残差头 tanh 100% 饱和（梯度基本没了）、注意力反向溢出成 NaN
            # （ScaledDotProductEfficientAttentionBackward0，step 24 起 66 个参数梯度同时非有限）。
            Q_MIN[0] = q.norm(dim=-1).min().detach()
            q = F.normalize(q, dim=-1)
            mem("gate_pair")
            if args.s_seamless and "sn" not in disable:
                flat = q.reshape(b, NPATCH * N_IN, self.ed)
                rt = torch.softmax(self.sn_router(flat), -1)
                inj = torch.zeros_like(flat)
                for kk in range(len(self.offs)):
                    nb = flat.gather(1, self.sn_idx[kk].view(1, -1, 1).expand(b, -1, self.ed))
                    inj = inj + rt[..., kk:kk + 1] * self.sn_lin[kk % len(self.sn_lin)](nb) * self.sn_mask[kk].unsqueeze(-1)
                q = q + self._gain(self.sn_gain) * inj.view(b, NPATCH, N_IN, self.ed)
                q = F.normalize(q, dim=-1)      # 注入一次归一化一次；只在入口归一化会被这里重新撑回尺度（实测 q.std 又回到 41）
                mem("seamless")
            if args.attn_heads and "attn" not in disable:
                H, hd = args.attn_heads, self.ed // args.attn_heads
                z = q.reshape(b * NPATCH, N_IN, self.ed)
                sp = lambda t: t.reshape(-1, N_IN, H, hd).transpose(1, 2)
                a = F.scaled_dot_product_attention(sp(self.attn_q(z)), sp(self.attn_k(z)), sp(self.attn_v(z)))
                a = self.attn_o(a.transpose(1, 2).reshape(-1, N_IN, self.ed))
                q = q + self._gain(self.attn_gain) * a.view(b, NPATCH, N_IN, self.ed)
                q = F.normalize(q, dim=-1)
                mem("attn")
            s = torch.tanh(self.delta_gain)
            # 残差头的上界：tanh 限幅 + 有界增益 ⇒ 初值下 v 逐位等于 x（闸5 第二条盯这个）。
            if args.cont_head == "rgb":
                raw = self.head(q).squeeze(-1)
                v = x + s * torch.tanh(raw)
            else:
                # YCbCr 支路：逐像素把同格的三个通道位特征池化成一个向量，残差加在 YCbCr 坐标上。
                # 上一版这里是 head_ycb(q) 逐通道位输出 3 个数，形状直接对不上后面的 reshape——那是假连接。
                qp = q.view(b, NPATCH, N_IN // 3, 3, self.ed).mean(3)
                raw = self.head_ycb(qp)
                v = ycb_to_rgb(rgb_to_ycb(x.view(b, NPATCH, N_IN // 3, 3)) + s * torch.tanh(raw)).reshape(b, NPATCH, N_IN)
            if not torch.is_grad_enabled():
                HEAD_RAW[0] = float(raw.abs().mean())     # 残差头进 tanh 前的幅度
                HEAD_SAT[0] = float((raw.abs() > 2.0).float().mean())   # |·|>2 ⇒ tanh 导数 sech²<0.10，梯度基本没了
                Q_STD[0] = float(q.std())
                PART_STD[0] = (float(vs.std()), float(ass.std()), float(pss.std()), float(h.std()))
            mem("head")
            if "cinv" not in disable:
                d = torch.exp(torch.tanh(self.cgate) * (prompt @ self.logm)).clamp(min=1e-3)
                v = v / d.view(b, 1, 1, 3).expand(b, NPATCH, N_IN // 3, 3).reshape(b, NPATCH, N_IN)
            if "copy" not in disable:
                gt = self.tgate(q).squeeze(-1)
                if args.gate_task:
                    gt = gt + self.tgate_task(task).view(b, 1, 1)
                g = torch.sigmoid(gt)
                if not torch.is_grad_enabled():
                    GATE_MEAN[0] = float(g.mean())
                    GATE_ROWS.append(g.detach().reshape(g.shape[0], -1))
                v = g * x + (1 - g) * v
            return v.clamp(0.0, 1.0).reshape(b, NPATCH * N_IN)
        finally:
            IN_MODEL_PATH[0] = 0


gen = I2ICont().to(DEVICE)
p7 = P7WordRouter2048(word_dim=256, inner_dim=256, heads=16, head_dim=16, max_len=128, num_groups=4).to(DEVICE)
MODEL_SET = {"gen": gen, "p7": p7}
P7_REGISTERED = 449939      # 本次实测登记值；p7 是被复用的连接器，形状一变这条就该响
P7_DEAD_REGISTERED = []   # 真实链路下 p7 全部参数都有梯度（上一版登记的是我仪器没调用 p7 造成的假读数，已纠正）
TRAINABLE_TOTAL = 0

# ═══════════════════ 仪器闸 8 条 ═══════════════════
def gate1_inventory():
    names = set()
    for mn, m in MODEL_SET.items():
        for n, _ in m.named_parameters():
            names.add(mn + "." + n)
    need = ["gen.val.proj.weight", "gen.val.radius", "gen.attr_proj.weight", "gen.pctx_proj.weight",
            "gen.input_proj.weight", "gen.gate_pair.weight", "gen.sn_router.weight", "gen.tgate.weight",
            "gen.logm", "gen.cgate", "gen.attn_q.weight",
            "gen.head.weight" if args.cont_head == "rgb" else "gen.head_ycb.weight", "gen.sn_gain", "gen.attn_gain",
            "gen.tgate_task.weight", "gen.patch_pos", "gen.inpatch_pos", "gen.h_scale", "gen.delta_gain",
            "gen.gain_val", "gen.gain_attr", "gen.gain_pctx"]
    miss = [n for n in need if n not in names]
    assert not miss, "模块全集缺件: %s" % miss
    return names


def gate2_dims(names):
    ed = args.embed_dim
    checks = {"gen.val.proj.weight": (ed, 2 * args.fourier_k + 1),
              "gen.attr_proj.weight": (ed, ATTR_DIM),
              "gen.input_proj.weight": (ed, 256 + ATTR_DIM),
              "gen.pctx_proj.weight": (ed, ed),
              "gen.gate_pair.weight": (ed, ed * 2),
              "gen.sn_router.weight": (len(gen.offs), ed),
              "gen.tgate.weight": (1, ed),
              "gen.logm": (NT, 3)}
    bad = {}
    for n, shape in checks.items():
        t = None
        for mn, m in MODEL_SET.items():
            d = dict(m.named_parameters())
            if n.startswith(mn + ".") and n.split(".", 1)[1] in d:
                t = d[n.split(".", 1)[1]]
        if t is None:
            bad[n] = "找不到"
        elif tuple(t.shape) != tuple(shape):
            bad[n] = "%s≠%s" % (tuple(t.shape), shape)
    assert not bad, "维度配错: %s" % bad
    w("[闸1-2] 模块全集齐、逐模块维度全对")


def gate3_param_identity():
    gen_got = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    manual = (gen.val.proj.weight.numel() + gen.val.radius.numel() + gen.attr_proj.weight.numel() +
              gen.pctx_proj.weight.numel() + gen.input_proj.weight.numel() +
              sum(p.numel() for p in gen.encoder.parameters()) + gen.patch_pos.numel() + gen.inpatch_pos.numel() +
              gen.h_scale.numel() + gen.gate_pair.weight.numel() + gen.delta_gain.numel() +
              gen.gain_val.numel() + gen.gain_attr.numel() + gen.gain_pctx.numel() +
              sum(m.weight.numel() for m in gen.sn_lin) + gen.sn_router.weight.numel() + gen.sn_gain.numel() +
              sum(getattr(gen, n).weight.numel() for n in ("attn_q", "attn_k", "attn_v", "attn_o")) + gen.attn_gain.numel() +
              gen.tgate.weight.numel() + gen.tgate_task.weight.numel() +
              (gen.task_e.weight.numel() + gen.sigma_e.weight.numel() if args.task_cond else 0) +
              (gen.prompt_e.weight.numel() + gen.cond_gain.numel() if args.prompt_cond else 0) +
              gen.logm.numel() + gen.cgate.numel() +
              (gen.head.weight.numel() if args.cont_head == "rgb" else gen.head_ycb.weight.numel()))
    assert manual == gen_got, "gen 手写清单与实际可训差 %d ⇒ 有件被重复构造或没登记" % (gen_got - manual)
    p7_got = sum(p.numel() for p in p7.parameters() if p.requires_grad)
    assert p7_got == P7_REGISTERED, "p7 可训参数 %d ≠ 登记值 %d ⇒ 复用的连接器被改过，必须重新登记" % (p7_got, P7_REGISTERED)
    global TRAINABLE_TOTAL
    TRAINABLE_TOTAL = gen_got + p7_got
    w("[闸3] gen 手写恒等式 %s 对上；p7 登记值 %s 对上；合计可训 %s" %
      (f"{gen_got:,}", f"{p7_got:,}", f"{TRAINABLE_TOTAL:,}"))


def zz(*shape, dtype=torch.float32):
    return torch.zeros(*shape, dtype=dtype, device=DEVICE)


def gate4_no_bin():
    assert not hasattr(gen, "bin_embed"), "bin_embed 还在 ⇒ 表示层没换干净"
    assert not hasattr(gen, "mixW") and not hasattr(gen, "bin_gain"), "档轴置换表还在 ⇒ 变形仍是离散形式"
    z = zz
    d = torch.rand(2, NPATCH * N_IN, device=DEVICE)
    before = to_bins_legacy.__doc__
    try:
        gen(d, z(2, 256), z(2, ATTR_DIM), z(2, NPATCH, 256), z(2, dtype=torch.long), z(2, dtype=torch.long), z(2, NT))
    except AssertionError as e:
        raise AssertionError("模型路径里出现了旧量化器: %s" % e)
    w("[闸4] 模型路径内无 bin_embed / 无档轴表 / 旧量化器调用 0 次")


def gate5_identity_at_init():
    z = zz
    d = torch.rand(2, NPATCH * N_IN, device=DEVICE)
    sv, at, pc = torch.randn(2, 256, device=DEVICE), torch.randn(2, ATTR_DIM, device=DEVICE), torch.randn(2, NPATCH, 256, device=DEVICE)
    tk, sg, pr = torch.randint(0, 3, (2,), device=DEVICE), torch.randint(0, 4, (2,), device=DEVICE), torch.rand(2, NT, device=DEVICE)
    with torch.no_grad():
        base = gen(d, sv, at, pc, tk, sg, pr)
        off = gen(d, sv, at, pc, tk, sg, pr, disable=("cond", "prompt", "cinv"))
        dv = float((base - off).abs().max())
        di = float((base - d).abs().max())
    assert dv < 1e-6, "零初始化下条件/提示词/逆算子仍在改输出 %.3e ⇒ 它们不是恒等起点" % dv
    assert di < 1e-6, ("第0步整体输出不等于退化输入（差 %.3e）⇒ 变换支路在初值上就往恒等输入上叠噪声，"
                       "copy 门会在头几步被 Adam 顶到饱和、把变换支路整条饿死（上一版实测：门=1.0000、"
                       "考试读数逐位等于地板 22.792%%）") % di
    w("[闸5] 第0步恒等成立：cond/prompt/cinv 不改输出 max|Δ|=%.2e，且整体输出逐位＝退化输入 max|Δ|=%.2e" % (dv, di))


def gate6_on_off_after_perturb():
    """手册教训：恒等初始化的通路，必须先扰动再做差分，用完逐位还原。"""
    saved = {n: p.detach().clone() for n, p in gen.named_parameters()}
    g = torch.Generator(device="cpu").manual_seed(7)
    d = torch.rand(2, NPATCH * N_IN, generator=g).to(DEVICE)
    sv, at, pc = torch.randn(2, 256, generator=g).to(DEVICE), torch.randn(2, ATTR_DIM, generator=g).to(DEVICE), \
        torch.randn(2, NPATCH, 256, generator=g).to(DEVICE)
    tk = torch.randint(0, 3, (2,), generator=g).to(DEVICE)
    sg = torch.randint(0, 4, (2,), generator=g).to(DEVICE)
    pr = torch.zeros(2, NT, device=DEVICE); pr[0, 0] = 1; pr[1, 2] = 1
    pd = dict(gen.named_parameters())
    with torch.no_grad():
        pre = gen(d, sv, at, pc, tk, sg, pr)          # 扰动前的同输入输出，用作还原基准
        for n in ("task_e.weight", "sigma_e.weight", "prompt_e.weight", "logm"):
            pd[n].normal_(0, .5)
        for nm in ("cond_gain", "cgate", "sn_gain", "attn_gain"):
            pd[nm].fill_(1.0)
        for m in gen.sn_lin:
            m.weight.normal_(0, .1)
        base = gen(d, sv, at, pc, tk, sg, pr)
        worst = {}
        for name in ("cond", "prompt", "sn", "attn", "cinv", "copy"):
            o = gen(d, sv, at, pc, tk, sg, pr, disable=(name,))
            worst[name] = float((base - o).abs().max())
        for n, p in saved.items():
            pd[n].data.copy_(p)
    for name, dv in worst.items():
        assert dv > 1e-5, "扰动后关掉 %s 输出只变 %.2e ⇒ 这条路是假连接" % (name, dv)
        w("      差分[%-6s] max|Δ|=%.4f" % (name, dv))
    with torch.no_grad():
        post = gen(d, sv, at, pc, tk, sg, pr)
        r = float((post - pre).abs().max())
        pr2 = max(float((pd[n].data - p).abs().max()) for n, p in saved.items())
    assert r < 1e-6 and pr2 < 1e-12, "扰动未逐位还原：输出差 %.3e 参数差 %.3e ⇒ 后面所有读数都不算" % (r, pr2)
    w("[闸6] 六条通路扰动后'关掉就变'，且已逐位还原（输出残差 %.1e，参数残差 %.1e）" % (r, pr2))


def gate7_grad_coverage():
    for p in list(gen.parameters()) + list(p7.parameters()):
        p.grad = None
    g = torch.Generator(device="cpu").manual_seed(11)
    d = torch.rand(2, NPATCH * N_IN, generator=g).to(DEVICE)
    sv, at, pc = torch.randn(2, 256, generator=g).to(DEVICE), torch.randn(2, ATTR_DIM, generator=g).to(DEVICE), \
        torch.randn(2, NPATCH, 256, generator=g).to(DEVICE)
    tk = torch.randint(0, 3, (2,), generator=g).to(DEVICE)
    sg = torch.randint(0, 4, (2,), generator=g).to(DEVICE)
    pr = torch.zeros(2, NT, device=DEVICE); pr[:, 0] = 1
    tgt = torch.rand(2, NPATCH * N_IN, generator=g).to(DEVICE)
    ptok = at.unsqueeze(1).expand(-1, NPATCH, -1).contiguous()
    wo, sv2, _, _, _ = p7.forward_batch(ptok, ptok, [NPATCH] * 2, [NPATCH] * 2, last_loss=0.0)
    (gen(d, sv2, at, wo, tk, sg, pr) - tgt).abs().mean().backward()
    allp = list(gen.named_parameters()) + [("p7." + k, v) for k, v in p7.named_parameters()]
    dead_gen = [n for n, p in gen.named_parameters() if p.requires_grad and p.grad is None]
    dead_p7 = sorted("p7." + n for n, p in p7.named_parameters() if p.requires_grad and p.grad is None)
    assert not dead_gen, "gen 里有参数没拿到梯度（登记了但空转）: %s" % dead_gen
    assert dead_p7 == P7_DEAD_REGISTERED, \
        "p7 空转参数集合变了：现在 %s，登记 %s ⇒ 连接器参与面被改动，必须重新核对" % (dead_p7, P7_DEAD_REGISTERED)
    n_tr = len([1 for _, p in allp if p.requires_grad])
    live = [n for n, q in p7.named_parameters() if q.requires_grad and "p7." + n not in dead_p7]
    dl = sum(q.numel() for n, q in p7.named_parameters() if "p7." + n in dead_p7)
    ll = sum(q.numel() for n, q in p7.named_parameters() if n in live)
    w("[闸7] gen 全 %d 个可训参数均有梯度；p7 参与 %d 个张量/%s 参数，空转 %d 个张量/%s 参数（空转占比 %.1f%%，已登记）"
      % (len([1 for _, q in gen.named_parameters() if q.requires_grad]), len(live), f"{ll:,}", len(dead_p7), f"{dl:,}", 100 * dl / max(1, dl + ll)))


def gate8_composition_exact():
    """连续域里组合精确：log(Πm)=Σlog m，残差必须为 0（对比 G14 的网格比 0.891）。"""
    worst = 0.0
    for a in range(NT):
        for b in range(NT):
            m1, m2 = MULT_TABLE[a], MULT_TABLE[b]
            worst = max(worst, float(np.abs(np.log(m1 * m2) - (np.log(m1) + np.log(m2))).max()))
    assert worst < 1e-6, "乘子组合不闭合，残差 %.3e" % worst
    tgt = np.log(np.array([.55, .7], np.float32))
    w("[闸8] 变形名组合精确闭合(残差 %.1e)；oracle 可直接解析给出 a*=%s，不存在'网格含不住目标'" % (worst, np.round(tgt, 4).tolist()))


def run_all_gates():
    names = gate1_inventory(); gate2_dims(names); gate3_param_identity(); gate4_no_bin()
    gate5_identity_at_init(); gate6_on_off_after_perturb(); gate7_grad_coverage(); gate8_composition_exact()
    w("[仪器闸] 8 条全过 ⇒ 允许开训")


run_all_gates()
if args.gates_only:
    w("[gates_only] 只跑仪器闸，退出"); sys.exit(0)
params = list(gen.parameters()) + list(p7.parameters())
NAMED = [("gen." + n, p) for n, p in gen.named_parameters()] + [("p7." + n, p) for n, p in p7.named_parameters()]
assert sum(p.numel() for p in params if p.requires_grad) == TRAINABLE_TOTAL, \
    "进入优化器的可训参数与闸3登记不符"
opt = torch.optim.Adam(params, lr=args.lr)
rng = np.random.default_rng(SEED)
EXAM_SRC = (exam_labeled if (args.smoke or args.exam_only_labeled) else exam_labeled + exam_generic)
EXAM = build_pairs(EXAM_SRC, 1, np.random.default_rng(SEED + 1), exam=True)
w(f"[考卷] {EXAM['deg'].shape[0]} 行（7 组合上色 + 3 任务基线 × {len(EXAM_SRC)} 图）")
EXAM_TASK = np.array([r[1] for r in EXAM["rows"]])   # 分任务尺子：总分掩盖上色，上一代就是这么翻车的


def batch_of(P, sl):
    n = len(sl)
    deg = torch.from_numpy(P["deg"][sl]).float().to(DEVICE)
    attr = torch.from_numpy(P["attr"][sl]).float().to(DEVICE)
    task = torch.tensor([P["rows"][j][1] for j in sl], dtype=torch.long, device=DEVICE)
    sig = torch.from_numpy(P["sig"][sl]).long().to(DEVICE)
    pr = torch.from_numpy(P["prompt"][sl]).float().to(DEVICE)
    tgt = torch.from_numpy(P["tgt"][sl]).float().to(DEVICE)
    ptok = attr.unsqueeze(1).expand(-1, NPATCH, -1).contiguous()
    wo, sv, _, _, _ = p7.forward_batch(ptok, ptok, [NPATCH] * n, [NPATCH] * n, last_loss=0.0)
    return deg, sv, attr, wo, task, sig, pr, tgt


def chroma_split(pred, tgt):
    p = pred.reshape(-1, 3); t = tgt.reshape(-1, 3)
    yp = .299 * p[:, 0] + .587 * p[:, 1] + .114 * p[:, 2]
    yt = .299 * t[:, 0] + .587 * t[:, 1] + .114 * t[:, 2]
    cbp = .5 - .168736 * p[:, 0] - .331264 * p[:, 1] + .5 * p[:, 2]
    cbt = .5 - .168736 * t[:, 0] - .331264 * t[:, 1] + .5 * t[:, 2]
    crp = .5 + .5 * p[:, 0] - .418688 * p[:, 1] - .081312 * p[:, 2]
    crt = .5 + .5 * t[:, 0] - .418688 * t[:, 1] - .081312 * t[:, 2]
    return float((cbp - cbt).abs().mean() + (crp - crt).abs().mean()), float((yp - yt).abs().mean())


@torch.no_grad()
def score(pred, tgt):
    p = pred.reshape(pred.shape[0], -1).cpu().numpy(); t = tgt.reshape(tgt.shape[0], -1).cpu().numpy()
    l1 = float(np.abs(p - t).mean()) * 100.0
    cs = [float(np.corrcoef(p[i], t[i])[0, 1]) for i in range(p.shape[0]) if p[i].std() > 1e-8 and t[i].std() > 1e-8]
    corr = float(np.median(cs)) if cs else float("nan")
    legacy = float((to_bins_legacy(p) == to_bins_legacy(t)).mean()) * 100.0
    return l1, corr, legacy


def exam_eval(tag):
    outs = []
    dg = []
    n = EXAM["deg"].shape[0]
    with torch.no_grad():          # 上一版漏了这行：outs 把每批的前向图全存住，19 批＝11.12GiB 直接撑爆卡
        GATE_ROWS.clear()
        for i0 in range(0, n, 2):
            sl = list(range(i0, min(i0 + 2, n)))
            deg, sv, attr, wo, task, sig, pr, tgt = batch_of(EXAM, sl)
            outs.append((gen(deg, sv, attr, wo, task, sig, pr), tgt))
            dg.append(deg)
    p = torch.cat([o[0] for o in outs]); t = torch.cat([o[1] for o in outs])
    dp = 0.0 if EXAM_PREV[0] is None else float((p - EXAM_PREV[0]).abs().max())
    EXAM_PREV[0] = p.clone()
    l1, corr, legacy = score(p, t)
    ch, yl = chroma_split(p, t)
    d = torch.cat(dg)
    floor_id = float((d - t).abs().mean()) * 100.0
    floor_const = float((torch.full_like(t, .5) - t).abs().mean()) * 100.0
    w("  [评测底] copy门均开度=%.4f 预测均幅=%.4f 与上次考试预测最大差=%.3e | 地板: 原样吐出deg=%.3f%% 常数0.5=%.3f%%" % (
        GATE_MEAN[0], float(p.abs().mean()), dp, floor_id, floor_const))
    w("  [支路] q.std=%.4f 残差头|raw|均=%.4f 饱和占比(|raw|>2)=%.4f 增益tanh(delta_gain)=%.4f delta_gain=%.4f" % (
        Q_STD[0], HEAD_RAW[0], HEAD_SAT[0], float(torch.tanh(gen.delta_gain)), float(gen.delta_gain)))
    w("  [尺度] 注入前裸值 std: val=%.3f attr_proj=%.3f pctx_proj=%.3f h=%.3f | 有界增益 2*σ(g): val=%.3f attr=%.3f pctx=%.3f h_scale=%.3f" % (
        tuple(list(PART_STD[0]) + [float(2.0 * torch.sigmoid(gen.gain_val)), float(2.0 * torch.sigmoid(gen.gain_attr)),
                                   float(2.0 * torch.sigmoid(gen.gain_pctx)), float(gen.h_scale)])))
    w(f"  [考{tag}] 像素L1={l1:.3f}% 比地板好={floor_id - l1:+.3f}pp corr={corr:.4f} 旧bin口径={legacy:.2f}% 色度L1={ch:.4f} 亮度L1={yl:.4f} n={p.shape[0]}")
    pn, tn, dn = p.cpu().numpy(), t.cpu().numpy(), d.cpu().numpy()
    ggm = torch.cat(GATE_ROWS, 0) if len(GATE_ROWS) else None
    for ti, tname in enumerate(TASKS):
        m = EXAM_TASK == ti
        if int(m.sum()) == 0:
            continue
        pm, tm, dm = pn[m], tn[m], dn[m]
        gtxt = "n/a" if ggm is None else "%.4f" % float(ggm[torch.from_numpy(m)].mean())
        f_ti = float(np.abs(dm - tm).mean()) * 100.0
        l_ti = float(np.abs(pm - tm).mean()) * 100.0
        lg_ti = float((to_bins_legacy(pm) == to_bins_legacy(tm)).mean()) * 100.0
        w("    [考%s/%s] L1=%.3f%% 地板=%.3f%% 比地板=%+.3fpp 旧bin口径=%.2f%% copy门开度=%s n=%d" % (
            tag, tname, l_ti, f_ti, f_ti - l_ti, lg_ti, gtxt, int(m.sum())))
    if steps >= args.brake_after and l1 >= floor_id:
        w("[刹车] 步数已过 %d，考试 L1 仍不比『原样吐出 deg』好（%.3f%% ≥ %.3f%%）⇒ 变换支路没在贡献任何东西，"
          "继续跑只会把分数记到恒等地板头上。本轮结论：不成立，停。" % (args.brake_after, l1, floor_id))
        sys.exit(5)
    w("  [参数指纹] head.sum=%.6f tgate.sum=%.6f sn_gain=%.6f logm|.|=%.6f steps=%d" % (
        float((gen.head if args.cont_head == "rgb" else gen.head_ycb).weight.sum()),
        float(gen.tgate.weight.sum()), float(gen.sn_gain), float(gen.logm.abs().sum()), steps))
    return l1, corr


if str(DEVICE).startswith("cuda"):
    torch.cuda.empty_cache()
    fr, tm = torch.cuda.mem_get_info()
    w("[显存] 开训前 卡总=%.2fGiB 空闲=%.2fGiB 本进程在活=%.2fGiB" % (tm / GB, fr / GB, torch.cuda.memory_allocated() / GB))
EP = 3 if args.smoke else args.epochs
if args.detect_anomaly:
    torch.autograd.set_detect_anomaly(True)
    w("[仪器] autograd 逐算子异常回溯已开（会明显变慢，只用于点名 NaN 来源）")
POOL = build_pairs(train_labeled, args.k_labeled, rng)
N = POOL["deg"].shape[0]
w(f"[训练池] {N} 对 | bs={args.batch_size} | 每轮步数={-(-N // args.batch_size)}")
_pd, _pt = torch.from_numpy(POOL["deg"]).float().to(DEVICE), torch.from_numpy(POOL["tgt"]).float().to(DEVICE)
w("[地板] 训练池 恒等=%.3f%% 常数0.5=%.3f%%（loss 若停在这两行中间就是没在学，别看相对跌幅）" % (
    float((_pd - _pt).abs().mean()) * 100, float((_pt - .5).abs().mean()) * 100))
_ptk = np.array([r[1] for r in POOL["rows"]])
w("       分任务恒等地板: " + " ".join("%s=%.3f%%" % (TASKS[ti], float((_pd[_ptk == ti] - _pt[_ptk == ti]).abs().mean()) * 100)
                                  for ti in range(3) if (_ptk == ti).sum()))
steps = 0
for ep in range(1, EP + 1):
    perm = rng.permutation(N)
    tot = chs = yls = 0.0; nb = 0
    for i0 in range(0, N if not args.smoke else min(N, 8), args.batch_size):
        sl = list(perm[i0:i0 + args.batch_size])
        if len(sl) < 2:
            continue
        deg, sv, attr, wo, task, sig, pr, tgt = batch_of(POOL, sl)
        opt.zero_grad(set_to_none=True)
        if steps == 0:
            PROBE["on"] = 1
            torch.cuda.reset_peak_memory_stats() if str(DEVICE).startswith("cuda") else None
        pred = gen(deg, sv, attr, wo, task, sig, pr)
        mem("fwd")
        loss = (pred - tgt).abs().mean()
        loss.backward()
        mem("bwd")
        PROBE["on"] = 0
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
        if not torch.isfinite(gn) or not torch.isfinite(loss):
            # NaN 止损闸：clip 遇到 inf 梯度会把系数算成 0，0*inf=NaN，一步就把整张量污染掉。
            # 所以必须在这里点名，而不是继续 opt.step()。
            badg = [n for n, p in NAMED if p.grad is not None and not torch.isfinite(p.grad).all()]
            badp = [n for n, p in NAMED if not torch.isfinite(p).all()]
            w("[NaN止损] step=%d loss=%.4g 梯度范数=%.4g | 梯度非有限的参数 %d 个: %s | 参数已被污染 %d 个: %s"
              % (steps + 1, float(loss), float(gn), len(badg), badg[:12], len(badp), badp[:12]))
            w("           min‖q‖=%.4e（F.normalize 的分母）| q.std=%.4e | 上一档 delta_gain=%.4e ⇒ 看分母有没有贴零" % (
                0.0 if Q_MIN[0] is None else float(Q_MIN[0]), Q_STD[0], float(gen.delta_gain)))
            sys.exit(6)
        opt.step()
        tot += float(loss); chs, yls = chroma_split(pred.detach(), tgt); nb += 1; steps += 1
    if steps == 0:
        w("[空转失败] 一步都没走，不写[完成]"); sys.exit(4)
    w(f"E{ep} loss={tot / max(1, nb):.5f} 色度L1={chs:.5f} 亮度L1={yls:.5f} 用时{time.time() - T0:.0f}s")
    if ep % (1 if args.smoke else args.exam_every) == 0:
        exam_eval(f"E{ep}")

w("[完成] 连续域引擎跑通，总步数 %d" % steps)
