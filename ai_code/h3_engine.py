# -*- coding: utf-8 -*-
"""H3 DiT 手写推理引擎（照 ltx_engine.py 的路子：逐层 pread + 手写算子，完全不用 diffusers pipeline）
   结构依据（读自 diffusers transformer_minimax_h3.py + checkpoint 实测形状）:
     50 层 · hidden 5376 · heads 56 · head_dim 128 · ffn 14336(swiglu 融合 28672) · patch (1,2,2)
     attn: qkv_proj[21504,5376] 切三份 -> to_q/k/v [7168,5376] · q_norm/k_norm[128] · out_proj[5376,7168]
     ff  : fc1[28672,5376]（gate+up 融合）· fc2[5376,14336]
     adaln: linear[96768,2688] (6*5376*3模态) · norm1/norm2[5376]
"""
import os, json, struct, math, threading, queue, time, sys as _sys
# ★ 自带文件日志（会话/终端死了日志仍在盘上 ✅ 主人 2026-10-14 令）
_LOG_PATH = os.environ.get("H3_LOG", r"D:\ai\h3_run.log")
try:
    _LOG_FH = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
except Exception:
    _LOG_FH = None
_orig_print = print
def print(*a, **k):                       # noqa: A001
    k.setdefault("flush", True)
    try:
        _orig_print(*a, **k)
    except UnicodeEncodeError:   # [FIX-1015 ENC-SAFE] GBK 控制台下 emoji 不再打死进程
        _orig_print(*[str(x).encode('ascii', 'replace').decode('ascii') for x in a], **k)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(" ".join(str(x) for x in a) + "\n")
        except Exception:
            pass
import torch, torch.nn.functional as F

DEV = "cuda"
DTYPE = torch.bfloat16
# 盘上即 F32、官方也保持 F32 的"精度岛"（diffusers _keep_in_fp32_modules / ComfyUI model.py:310）
_FP32_KEYS = ("video_patch_proj", "audio_patch_proj", "time_embedder",
              "final_layer.video_out", "final_layer.audio_out", "rope")
H3 = r"D:\ai\models\MiniMax-H3\Ref2VA"
TF = os.path.join(H3, "transformer")
MODE_NUM = 3          # token_tags: 0 video, 1 text, 2 audio
HEAD_DIM = 128
HEADS = 56
HID = 5376

_hdr = {}
_hlen = {}
def _h(sp):
    h = _hdr.get(sp)
    if h is None:
        with open(sp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        _hdr[sp] = h; _hlen[sp] = n          # [FIX-1015] 头长度一起缓存 => pread 不再多开一次文件
    return h

_ESIZE = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1}
_TDT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

def pread(name, sp, dtype=DTYPE, dev=None):
    h = _h(sp); m = h[name]; shape = m["shape"]; dt = m["dtype"]
    off = 8 + _hlen[sp] + m["data_offsets"][0]
    n = 1
    for x in shape: n *= x
    es = _ESIZE.get(dt) or _TDT[dt].itemsize
    buf = bytearray(n * es)                   # [FIX-1015] readinto: 一次分配, 不再 bytearray(bytes) 多拷一遍
    with open(sp, "rb") as f:
        f.seek(off); f.readinto(buf)
    out = torch.frombuffer(buf, dtype=_TDT.get(dt, torch.float32)).reshape(shape)
    return out.to(device=dev, dtype=dtype) if dev else out.to(dtype)

TURBO_PATHS = {"4": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
               "8": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"}
_LORA = {}
def load_turbo(path):
    _LORA.clear()          # ★ 每次只装一个（4 步 / 8 步 分别跑 ✅）
    """把 Turbo LoRA 合并进层权重（W' = W + (alpha/rank)·B@A ✅ comfyui 格式 ✅）"""
    from safetensors.torch import load_file
    sd = load_file(path)
    _mult = float(os.environ.get("H3_LORA_MULT", "1"))
    _last = 0.0
    mods = {}
    for k in sd:
        if ".lora_A.weight" in k:
            base = k.replace(".lora_A.weight", "")
            mods.setdefault(base, {})["A"] = sd[k]
        elif ".lora_B.weight" in k:
            mods.setdefault(k.replace(".lora_B.weight", ""), {})["B"] = sd[k]
        elif k.endswith(".alpha"):
            mods.setdefault(k[:-6], {})["alpha"] = sd[k]
    for base, d in mods.items():
        if "A" in d and "B" in d:
            A, B = d["A"], d["B"]                    # ★ 只留原始张量（bf16 ✅ 共 ~1.9GB ✅）
            # [FIX-1017 SWIGLU-NOSWAP] 撤回 FIX-1016 的两半对调。本文件元数据（一手）：
            #   swi_glu_mapping = "Diffusers [value;gate] -> ComfyUI [gate;value]"
            #   target_format   = "ComfyUI generic LoRA"
            #   ⇒ 盘上 B 的两半【已经是 [gate; value]】，与基座 fc1 同序（基座也是原生 [gate;value]，
            #     见 block_forward 的 SWIGLU-NATIVE 三证）。旧代码按 port 的 [value;gate] 读基座，
            #     于是又把 LoRA 对调一次 —— 两处一起错，正好互相"对上"，但都不是参考实现。
            al = float(d["alpha"].reshape(-1)[0]) if "alpha" in d else A.shape[0]
            rank = A.shape[0]
            # [FIX-1017 LORA-SCALE] 倍率回到 1（即 s = alpha/rank），一手证据：
            #   ① ComfyUI comfy/weight_adapter/lora.py:47  "scale = self.alpha / self.rank"
            #      （:20-24 rank = lora_A.shape[0]，diff = lora_up @ lora_down = B@A）
            #   ② 本文件实测 alpha：156 个模块 = 8.0、52 个 = 24.0（正是 50 blocks + 2 refiner 的融合 qkv），
            #      rank = A.shape[0] = 128 / 384 ⇒ alpha/rank 每个模块都恰好 0.0625
            #   ③ 元数据 training_alpha=8.0、training_rank=128、training_scale=0.0625 = 8/128，
            #      以及 qkv_fusion="block diagonal B; concat A; alpha multiplied by 3"
            #      —— alpha 乘 3 正是为了让 alpha/rank 在 rank×3 后仍等于 0.0625 ⇒ 缩放由加载器负责，
            #      B 里没有预先折进 training_scale。（FIX-1017 早先的 ×16 是我按"B 偏小"猜的，撤回。）
            #   可用 H3_LORA_MULT 覆盖（16 = 之前那次误判的口径）。
            _LORA[base] = (A, B, al / rank * _mult)   # ★ 不预计算 ✅ 用到哪层才旁路 ✅（省几十 GB ✅）
            _last = al / rank * _mult
    print("[e] Turbo LoRA 已载入: %d 个模块 ✅（原始 A/B ✅ 运行时旁路 ✅ 不预计算 ⇒ 省内存 ✅）"
          "★每模块强度 s = alpha/rank×%g = %.6f" % (len(_LORA), _mult, _last), flush=True)
    return len(_LORA)

_LORA_MAP = {"attn.qkv_proj": "attn.qkv_proj.weight", "attn.out_proj": "attn.out_proj.weight",
             "mlp.fc1": "mlp.fc1.weight", "mlp.fc2": "mlp.fc2.weight"}

def _add_lora(y, h, side, suf):
    """PEFT/ComfyUI 语义的运行时低秩旁路: y += s·B(Ax)。
       [FIX-1017 LORA-SIDE] 不再把增量并进权重：本 turbo 的增量约权重的 0.03%，
       而 bf16 一个 ulp = 0.39% ⇒ 合并加法被舍入整体吃掉（实测加与不加逐位相同），
       4/8 步蒸馏等于没装。旁路以完整幅度进入输出，与权重精度无关。"""
    e = side.get(suf) if side else None
    if e is None:
        return y
    A, B, s = e
    return y + F.linear(F.linear(h, A), B) * s

_LORA_GPU, _LORA_ORDER = {}, []
def lora_side(i, dev=DEV):
    """第 i 层的旁路张量上卡并 LRU 缓存（8 层 ≈ 300MB 显存）。"""
    if not _LORA:
        return None
    if i not in _LORA_GPU:
        d = {}
        for suf in _LORA_MAP:
            e = _LORA.get("diffusion_model.blocks.%d.%s" % (i, suf))
            if e is not None:
                A, B, s = e
                d[suf] = (A.to(dev), B.to(dev), s)
        _LORA_GPU[i] = d
        _LORA_ORDER.append(i)
        while len(_LORA_ORDER) > 8:
            _LORA_GPU.pop(_LORA_ORDER.pop(0), None)
    return _LORA_GPU[i]

_LORA_REF = None
def refiner_lora(dev=DEV):
    global _LORA_REF
    if not _LORA:
        return None
    if _LORA_REF is None:
        d = {}
        for i in (0, 1):
            for suf in _LORA_MAP:
                e = _LORA.get("diffusion_model.token_refiner.blocks.%d.%s" % (i, suf))
                if e is not None:
                    A, B, s = e
                    d[(i, suf)] = (A.to(dev), B.to(dev), s)
        _LORA_REF = d
        print("[e] token_refiner LoRA 旁路 %d/8 组 ✅" % len(d), flush=True)
    return _LORA_REF

class Reader:
    """逐层权重读取器（照 ltx_engine 的 PreadLayerReader ✅）"""
    def __init__(self):
        wm = json.load(open(os.path.join(TF, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
        self.layers, self.other = {}, {}
        for n, s in wm.items():
            if n.startswith("blocks."): self.layers.setdefault(int(n.split(".")[1]), []).append((n, os.path.join(TF, s)))
            else: self.other[n] = os.path.join(TF, s)
        self.n = max(self.layers) + 1
    def layer(self, i, dev="cpu"):
        st = {}
        for n, sp in self.layers[i]:
            rel = n.split(".", 2)[2]
            st[rel] = pread(n, sp, dev=dev)   # ★ 默认读进 CPU 内存（缓存 ✅）· 只有当前层再进 GPU 槽位 ✅
        return st
    def small(self, dev=DEV):
        # [FIX-1016 FP32-ISLAND] 官方 _keep_in_fp32_modules（transformer_minimax_h3.py:446）与
        #   ComfyUI native（model.py:310 注释 "output heads are the checkpoint's fp32 island"）一致：
        #   两个 patch 投影、time_embedder、两个输出头在盘上就是 F32 且官方保持 F32。
        #   旧代码统一降 bf16，等于把官方钉住的精度岛也削了。
        return {n: pread(n, sp, dtype=torch.float32 if n.startswith(_FP32_KEYS) else DTYPE).to(dev)
                for n, sp in self.other.items()}

def rms(x, w, eps=1e-5):
    # [FIX-1017 RMS-FP32] 两个官方都在 fp32 里算方差，旧代码整条在 bf16 里算：
    #   diffusers normalization.py:553-561: "variance = hidden_states.to(torch.float32).pow(2).mean(-1,keepdim=True)"
    #     → x*rsqrt(variance+eps) → 降到 weight 的 dtype → *weight；
    #   ComfyUI comfy/rmsnorm.py:7-11 走 torch.nn.functional.rms_norm（半精度内部升 fp32）。
    #   norm1/norm2/q_norm/k_norm/final_norm/refiner 全都过这里 ⇒ 50 层×4 个 norm 的 0.4% 级误差，白送的精度。
    v = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(v + eps)).to(w.dtype) * w

def lin(x, w, b=None):
    return F.linear(x, w, b)

# [FIX-1017 SWIGLU-AB] fc1 两半的顺序：两个官方口径相反，一度与实测冲突，最后由 2×2 A/B 裁定。
#   文档口径（native，ComfyUI ops.py:950-952 + port 的 LoRA 转换器注释 lora_conversion_utils.py:3301-3304
#     + port 的基座转换器 single_file_utils.py:4291-4294 "The checkpoint fuses `[gate; value]`,
#     `SwiGLU` reads `[value; gate]`"）：盘上 [gate;value]，算 fc2(silu(前半)*后半)。
#   port 口径（activations.py:143-146）：[value;gate]，算 前半*silu(后半)。
#   ★ 中途的假象：只改 SwiGLU（qkv 仍按 stacked 切）时，native 口径出片塌成灰
#     （|v|std 1.33→0.62，目标 xt std 0.337，帧 mean=75 std=6.5），port 口径反而"能认出参考图但乳白"
#     ⇒ 我一度以为文档错了。真相是 **qkv 行序那个更大的坑把两者一起压住了**（见下面 QKV-LAYOUT 的实测表）：
#     qkv 修对之后 native 口径 rel=0.0579、port 口径明显更差 ⇒ 文档口径从头就是对的。
#   教训：一次只改一个约定，且必须有"单次前向 + 有标准答案"的裁判（h3_score.py 的 rel），
#     不能用出片好坏反推约定——两个大坑叠在一起时，出片会给出相反的排序。
#   H3_SWIGLU=native|port 可切，默认 native。
_SWIGLU_NATIVE = os.environ.get("H3_SWIGLU", "native").strip().lower() != "port"

def _swiglu(a, g):
    """fc1 融合输出 chunk(2) 后的两半 -> FFN 中间激活。"""
    global _SWIGLU_SAID
    if not _SWIGLU_SAID:
        _SWIGLU_SAID = True
        print("[e] ★ SwiGLU 口径 = %s（H3_SWIGLU 可切） · ★ 融合 qkv 行序 = %s（H3_QKV 可切）" % (
            "native: silu(前半)*后半" if _SWIGLU_NATIVE else "port: 前半*silu(后半)",
            "interleaved: 逐头 [q k v] ⇒ 输出端重排" if _QKV_INTERLEAVED else "stacked: 直接切三段"), flush=True)
    return F.silu(a) * g if _SWIGLU_NATIVE else a * F.silu(g)

_SWIGLU_SAID = False

# [FIX-1017 QKV-LAYOUT] 融合 qkv 的行序有两种，键名与形状完全一样，只能靠来源判定：
#   py_h3/diffusers/loaders/single_file_utils.py:4276-4278 原文：
#     "Published single files store the reference model's post-load `[q; k; v]` stack;
#      **the MiniMaxAI/MiniMax-H3 shards interleave the rows per head, `[head0: q k v, head1: q k v, ...]`**.
#      Keys and shapes are identical, so the caller has to say which one it is."
#   同库 lora_conversion_utils.py:3167-3169："DiffSynth-Studio runs the raw checkpoint's per-head-interleaved
#     fused QKV verbatim, where every other producer reorders it to `[q_all; k_all; v_all]` at load time
#     **as the reference implementation does**" ⇒ 原生参考实现在加载时就把它 de-interleave 成 [q;k;v]，
#     ComfyUI 的 `qkv_proj(x).split(inner, -1)`（model.py:174-176）是直接切三段 ⇒ Comfy-Org 的整包文件已是 stacked。
#   我这份是 **MiniMaxAI/MiniMax-H3 的 13 个 shards**（有 model.safetensors.index.json），按上面原文属于 interleaved。
#   旧代码直接切三段 = 把 head0 的 k/v 当成 head1/head2 的 q 用 ⇒ 注意力全乱。
#   ★ 不改权重本体（231MB/层，逐层重排太贵），只在输出上按固定 index 重排；LoRA 的 B 是 stacked
#     （元数据 qkv_fusion="block diagonal B; concat A"，转换器对非 DiffSynth 文件直接 chunk(3)），
#     所以旁路增量必须在重排【之后】再加。H3_QKV=stacked|interleaved 可切。
#   ★★ 实测裁定（h3_score.py，σ=0.2，同一参考图/同一 cond/同一种子，rel 越小越好，1.0=啥也没预测）：
#        stacked     + native swiglu : rel=1.5904   |v_pred|std=0.694
#        stacked     + port   swiglu : rel=1.0910   |v_pred|std=0.411
#        interleaved + native swiglu : **rel=0.0579**  |v_pred|std=1.466（真值 |v*|std=1.485 ⇒ 98.7%）
#      且此时"官方 data-ward 速度"这一解释比第二名（-vo，3.8918）好 67 倍 ⇒ 输出约定也同时被证实。
#      ⇒ 默认 interleaved（文档口径与实测口径一致），旧的 stacked 只作对照保留。
_QKV_INTERLEAVED = os.environ.get("H3_QKV", "interleaved").strip().lower() != "stacked"
_QKV_PERM = None
if _QKV_INTERLEAVED:
    _w = torch.arange(3).view(3, 1, 1)
    _h_i = torch.arange(HEADS).view(1, HEADS, 1)
    _r = torch.arange(HEAD_DIM).view(1, 1, HEAD_DIM)
    _QKV_PERM = (_h_i * 3 * HEAD_DIM + _w * HEAD_DIM + _r).reshape(-1).to(DEV)

def _qkv_fix(y):
    """把 interleaved 盘序的融合 qkv 输出重排成 [q_all; k_all; v_all]。"""
    if _QKV_PERM is None:
        return y
    return y.index_select(-1, _QKV_PERM)


def rope_apply(x, cos, sin):
    """照 diffusers _apply_rotary_emb 复刻: 只转前 rotary_dim 维, rotate_half 约定 ✅
       x:(B,L,H,D) cos/sin:(L,rotary_dim)"""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    c = cos.to(x.dtype)[None, :, None, :]
    s2 = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = xr.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return torch.cat((xr * c + rot * s2, xp), dim=-1).contiguous()

# ============================================================================
# [FIX-1016 ROPE-POS] 位置网格：原样移植官方 diffusers 实现，逐比特对齐。
#   来源 D:\ai\py_h3\diffusers\modular_pipelines\minimax_h3\before_denoise.py:36-107
#   官方注释写死 "the float64 grid has to be reproduced exactly"，且 np.linspace(endpoint=False)
#   与 torch.linspace 结果不同 ⇒ 必须用 numpy + float64 造网格，不能自己写整数 arange。
#   旧代码 h3_engine.py:488-492 用 torch.long 的整数 t/h/w（空间 0..15、时间均匀 +1、文本全 0），
#   空间尺度比官方小 2 倍且未居中 ⇒ 出片呈 2 像素周期栅格（= patch 2×2）。
# ============================================================================
import numpy as _np

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float):
    """一根空间轴：dim//patch 个坐标，按宽高比在单位区间内居中，再乘 32。右端点不含 ⇒ 方形画布铺 [0,32)。"""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = _np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float):
    """每个 latent 帧的旋转时间，起点 origin。步长非均匀：5/3 * (1,4,4,4,4)。"""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    """一个 latent 帧内所有行的 (h, w) 旋转坐标，以及它所依据的宽度轴。"""
    sqrt_area = _np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(position_ids, rows, num_audio_latents, rotary_time, width_grid, audio_channels):
    """[FIX-1016 AUDIO-STREAM] 原样移植官方 before_denoise.py:112-133。
       音频块是 channel-major：同一时刻的两个声道行共享旋转时间，高度轴恒 0，
       宽度轴钉在【自己那块】width_grid 的两端（第一声道取首值、第二声道取末值）。"""
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])


def build_rope(position_ids, inv_freq, theta=10000.0):
    """position_ids: (L,3) float64 -> cos/sin (L,96)。
    官方 MiniMaxH3RotaryPosEmbed: 三轴各 rope_freq_dim=16 → cat 成 48 → 再自拼成 96，
    _apply_rotary_emb 只转前 96 维、head_dim 剩下 32 维直通（不是注释里以前写的 64）。"""
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)   # (L,3,16)
    f = freqs.reshape(freqs.shape[0], -1)                                            # (L,48) 轴序 t,h,w
    emb = torch.cat((f, f), dim=-1)[:, :HEAD_DIM]                                    # (L,96)
    return emb.cos(), emb.sin()

def _sdpa(q, k, v):
    """★ 显式用 flash attention kernel（大序列提速明显 ✅ 显存更省 ✅）"""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v)
    except Exception:
        return F.scaled_dot_product_attention(q, k, v)

def block_forward(x, st, temb_mod, adaln_idx, cos, sin, attn_mask=None, lora=None):
    """一层: adaln -> attn -> 残差 -> adaln -> ffn -> 残差（依据 diffusers 实现 ✅）
       [FIX-1016 ADALN-AXIS] temb_mod 的 6 张表现在是官方形状 (n_t*MODE_NUM, HID)：
       行序 [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]，按 idx=timestep_indices*MODE_NUM+token_tags
       沿 dim0 取行后广播到序列维。旧代码把表当 (n_t, 6, MODE_NUM, HID) 组优先切，
       并在 dim1(长度=MODE_NUM) 上索引 —— 只在 n_t==1 时不越界，且 6 组 shift/scale/gate 全部错位。"""
    sh1, sc1, g1, sh2, sc2, g2 = temb_mod
    idx = adaln_idx
    sh1 = sh1.index_select(0, idx).unsqueeze(0); sc1 = sc1.index_select(0, idx).unsqueeze(0)
    g1 = g1.index_select(0, idx).unsqueeze(0)
    sh2 = sh2.index_select(0, idx).unsqueeze(0); sc2 = sc2.index_select(0, idx).unsqueeze(0)
    g2 = g2.index_select(0, idx).unsqueeze(0)
    h = rms(x, st["norm1.weight"])
    h = h * (1.0 + sc1) + sh1                                                  # ★ 官方模态优先表
    qkv = _add_lora(_qkv_fix(lin(h, st["attn.qkv_proj.weight"])), h, lora, "attn.qkv_proj")
    q = qkv[:, :, :7168].view(h.shape[0], -1, HEADS, HEAD_DIM)
    k = qkv[:, :, 7168:14336].view(h.shape[0], -1, HEADS, HEAD_DIM)
    v = qkv[:, :, 14336:].view(h.shape[0], -1, HEADS, HEAD_DIM)
    q = rms(q, st["attn.q_norm.weight"]); k = rms(k, st["attn.k_norm.weight"])
    q = rope_apply(q, cos, sin); k = rope_apply(k, cos, sin)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = _sdpa(q, k, v)
    o = o.transpose(1, 2).reshape(x.shape[0], -1, HEADS * HEAD_DIM)
    o = _add_lora(lin(o, st["attn.out_proj.weight"]), o, lora, "attn.out_proj")
    x = x + g1 * o
    h = rms(x, st["norm2.weight"])
    h = h * (1.0 + sc2) + sh2
    a, g = _add_lora(lin(h, st["mlp.fc1.weight"]), h, lora, "mlp.fc1").chunk(2, dim=-1)
    # [FIX-1017 SWIGLU-NATIVE] silu 打在【前半】= 原生 [gate; value] 序。三处一手证据：
    #   ① py_h3/diffusers/loaders/lora_conversion_utils.py:3301-3304 原文
    #      "the reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]` while `SwiGLU`
    #       computes `value * silu(gate)` from a fused `[value; gate]`, so the two halves swap places"
    #      —— port 加载原生权重时【要交换两半】，说明盘上就是 [gate; value]。
    #   ② ComfyUI comfy/ops.py:950-952 _swiglu_eager: "gate, up = x.chunk(2,-1); silu(gate).mul_(up)"。
    #   ③ transformer/config.json 的字段名（rope_inv_freq_len / adaln_out_features / latents_dim /
    #      token_refiner_num_layers / timestep_input_dim）与 ComfyUI MiniMaxH3Model.__init__(model.py:475-480)
    #      逐字对应，而 port 用的是另一套名字（freq_dim/in_channels/num_refiner_layers）⇒ 这份权重是原生格式。
    #   ★ FIX-1016 SWIGLU-HALF 只读了 port 的 activations.py:145 就把 silu 挪到后半，等于 50 层 + 2 层 refiner
    #     的 FFN 全在算另一个函数 —— 那是我自己引入的缺陷，此处撤回。
    #   ★ 撤回后单独跑曾塌成灰，一度让我以为文档错了；真凶是融合 qkv 的逐头交错行序（见 QKV-LAYOUT），
    #     两个坑叠在一起时出片会给出相反排序。qkv 修对后 native 口径 rel=0.0579 ⇒ 文档口径成立。
    h = _swiglu(a, g)
    h = _add_lora(lin(h, st["mlp.fc2.weight"]), h, lora, "mlp.fc2")
    return x + g2 * h

def auto_slice(reader, resident=3, cache=6, prefetch=True):
    """[FIX-1015] 流式前向。三处修复：
       ① 预取线程只暖 CPU 缓存，H2D 与 LoRA 合并只在主线程做 => 不再与主线程抢同一个 GPU 槽位(数据竞争)
       ② CPU 缓存存【原始】权重，LoRA 在 GPU 上按 bf16 合并 => 去掉 CPU float32 大矩阵乘
       ③ 常驻层也合并 LoRA（旧版漏了 => 前 N 层是半 Turbo）；内存守卫带日志、阈值可调"""
    _cache, _order = {}, []
    _inflight = {}
    _lk = threading.Lock()
    _slot = {}
    GUARD_MB = float(os.environ.get("H3_GUARD_MB", "2500"))
    def _free_mb():
        try:
            import ctypes
            class _M(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _M(); m.dwLength = ctypes.sizeof(_M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1048576.0
        except Exception:
            return 99999.0
    def _evict(i):
        while len(_order) > cache:
            old = _order.pop(0)
            if old != i: _cache.pop(old, None)
        if _free_mb() < GUARD_MB:
            fm = _free_mb()
            while len(_order) > 2 and _free_mb() < GUARD_MB:
                old = _order.pop(0)
                if old != i: _cache.pop(old, None)
            print("[e] 内存守卫: 可用 %.0fMB < %.0fMB => 缓存降到 %d 层(将重读磁盘)" % (fm, GUARD_MB, len(_order)), flush=True)
    def _cpu(i):
        with _lk:
            hit = _cache.get(i)
        if hit is not None:
            return hit
        fresh = reader.layer(i, dev="cpu")
        with _lk:
            if _cache.get(i) is None:
                _cache[i] = fresh; _order.append(i); hit = fresh
            else:
                hit = _cache[i]
            _evict(i)
        return hit
    _resident = {}
    for _i in range(min(resident, reader.n)):
        try:
            _st = reader.layer(_i, dev=DEV)
            _resident[_i] = _st
        except Exception as e:
            print("[e] 常驻层 %d 失败: %r" % (_i, e), flush=True)
            break
    print("[e] 常驻 GPU: %d 层%s · 显存 %.2fGB · 可用内存 %.0fMB" % (
        len(_resident), "（含 LoRA 已合并）" if _LORA else "（未挂 LoRA）",
        torch.cuda.memory_allocated()/1024**3, _free_mb()), flush=True)
    def _get(i):
        if i in _resident:
            return _resident[i]
        hit = _cpu(i)
        for k, v in hit.items():
            sl = _slot.get(k)
            if sl is None or sl.shape != v.shape:
                _slot[k] = v.to(DEV, non_blocking=True).clone()
            else:
                _slot[k].copy_(v, non_blocking=True)
        # [FIX-1015c] 不在此处合并: 权重已在 CPU 缓存里合并好, 避免每步 GPU 临时张量把显存推到上限
        return _slot
    if prefetch:
        def _worker(i):
            try:
                _cpu(i)
            except Exception:
                pass
            finally:
                with _lk:
                    _inflight.pop(i, None)
    def layer(i):
        st = _get(i)
        if prefetch and (i + 1) < reader.n and (i + 1) not in _resident:
            t = None
            with _lk:
                if _inflight.get(i + 1) is None:
                    t = threading.Thread(target=_worker, args=(i + 1,), daemon=True)
                    _inflight[i + 1] = t
            if t is not None:
                t.start()
        return st
    return layer

def time_embed(t, dim=256, flip=True, shift=0.0, scale=1):
    """复刻 diffusers Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - shift))
    args = t.to(torch.float32)[:, None] * scale * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1) if flip else emb

def refiner(t, st, sm, lora=None):
    """token_refiner: 2 层 block + final_norm（结构与主 block 同 ✅；LoRA 旁路键为 (层号, 模块名)）"""
    for i in (0, 1):
        p = "token_refiner.blocks.%d." % i
        side = dict((k[1], v) for k, v in lora.items() if k[0] == i) if lora else None
        x = rms(t, sm[p + "norm1.weight"])
        qkv = _add_lora(_qkv_fix(lin(x, sm[p + "attn.qkv_proj.weight"])), x, side, "attn.qkv_proj")
        q = qkv[:, :, :7168].view(t.shape[0], -1, HEADS, HEAD_DIM)
        k = qkv[:, :, 7168:14336].view(t.shape[0], -1, HEADS, HEAD_DIM)
        v = qkv[:, :, 14336:].view(t.shape[0], -1, HEADS, HEAD_DIM)
        q = rms(q, sm[p + "attn.q_norm.weight"]); k = rms(k, sm[p + "attn.k_norm.weight"])
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        o = o.transpose(1, 2).reshape(t.shape[0], -1, HEADS * HEAD_DIM)
        o = _add_lora(lin(o, sm[p + "attn.out_proj.weight"]), o, side, "attn.out_proj")
        t = t + o
        x = rms(t, sm[p + "norm2.weight"])
        a, g = _add_lora(lin(x, sm[p + "mlp.fc1.weight"]), x, side, "mlp.fc1").chunk(2, dim=-1)
        hh = _swiglu(a, g)                    # [FIX-1017 SWIGLU-AB] 与 block_forward 同一开关
        t = t + _add_lora(lin(hh, sm[p + "mlp.fc2.weight"]), hh, side, "mlp.fc2")
    return rms(t, sm["token_refiner.final_norm.weight"])

def dit_forward(reader, x_video, x_audio, cond, timestep, timestep_indices, token_tags, position_ids,
                video_indices, audio_indices, text_indices, sm=None, layer_fn=None, verbose=False):
    """一次 DiT 前向（依据 diffusers 实现逐行复刻 ✅ 全 GPU 计算 ✅）"""
    t0 = time.perf_counter()
    sm = sm or reader.small()
    layer_fn = layer_fn or auto_slice(reader, resident=3, cache=6)
    inv = sm["rope.inv_freq"].to(torch.float32)
    # [FIX-1016 FP32-ISLAND] 两个 patch 投影按权重 dtype（fp32）算，进打包缓冲时降到文本流 dtype
    #   （官方 transformer_minimax_h3.py:630-638：输入对齐投影 dtype，缓冲 dtype 由文本流定）
    ve = lin(x_video.to(DEV, sm["video_patch_proj.weight"].dtype), sm["video_patch_proj.weight"], sm["video_patch_proj.bias"])
    ae = lin(x_audio.to(DEV, sm["audio_patch_proj.weight"].dtype), sm["audio_patch_proj.weight"], sm["audio_patch_proj.bias"])
    te = lin(cond.to(DEV, DTYPE), sm["condition_proj.weight"], sm["condition_proj.bias"])
    te = refiner(te, None, sm, refiner_lora())
    L = position_ids.shape[0]
    h = te.new_zeros((te.shape[0], L, te.shape[-1]))
    h = h.index_copy(1, text_indices.to(DEV), te)
    h = h.index_copy(1, video_indices.to(DEV), ve.to(te.dtype))
    h = h.index_copy(1, audio_indices.to(DEV), ae.to(te.dtype))
    # [FIX-1016 TEMB-DTYPE] 官方: AdaLN 各自 "先在自己的高精度下算 silu，再降到该投影权重的 dtype"。
    #   本 checkpoint 的 time_embedder.proj_* 在盘上就是 bf16（官方 _keep_in_fp32_modules 是加载期行为，
    #   我们的 pread 不做 dtype 转换），所以这里让线性层跟着权重 dtype，但 silu 一律在 fp32 里算完再降。
    # [FIX-1016 FP32-ISLAND] time_embedder 整条 fp32（官方 643-644：temb 留在 time_embedder 的精度，
    #   各 AdaLN 自己做激活后再降到各自投影的 dtype）；旧代码把权重降 bf16 再算
    temb = time_embed(timestep.to(DEV))
    temb = lin(temb, sm["time_embedder.proj_in.weight"], sm["time_embedder.proj_in.bias"])
    temb = lin(F.silu(temb), sm["time_embedder.proj_out.weight"], sm["time_embedder.proj_out.bias"])
    idx = (timestep_indices.to(DEV) * MODE_NUM + token_tags.to(DEV)).to(torch.long)
    cos, sin = build_rope(position_ids.to(DEV), inv)
    for i in range(reader.n):
        st = layer_fn(i)
        # [FIX-1016 ADALN-AXIS] 官方: linear(silu(temb)) -> view(-1, 6*HID) -> chunk(6, dim=-1)
        #   ⇒ 6 张 (n_t*MODE_NUM, HID) 表，行序 [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]（模态优先）
        _aw = st["adaln_proj.linear.weight"]
        a = lin(F.silu(temb.float()).to(_aw.dtype), _aw, st["adaln_proj.linear.bias"]).view(-1, 6 * HID)
        sh1, sc1, g1, sh2, sc2, g2 = a.chunk(6, dim=-1)
        h = block_forward(h, st, (sh1, sc1, g1, sh2, sc2, g2), idx, cos, sin, lora=lora_side(i))
        if verbose and (i % 5 == 0 or i == reader.n - 1):
            print("[e] layer %d/%d  %.1fs  显存 %.2fGB" % (i, reader.n, time.perf_counter() - t0, torch.cuda.memory_allocated()/1024**3), flush=True)
    hh = rms(h, sm["final_layer.norm.weight"])
    # 末层: 官方 linear 只出 2*HID（不带模态维），按 timestep_indices 沿 dim0 取行
    _fw = sm["final_layer.adaln_proj.linear.weight"]
    ao = lin(F.silu(temb.float()).to(_fw.dtype), _fw, sm["final_layer.adaln_proj.linear.bias"]).view(-1, 2 * HID)
    # [FIX-1016 OUT-ORDER] 官方 MiniMaxH3AdaLayerNormOut.forward(transformer_minimax_h3.py:151):
    #   shift, scale = linear(...).chunk(2, dim=-1) —— 前半是 shift、后半是 scale（与 LTX2/Wan 输出层同序）。
    #   旧代码写成 sc, sh = chunk(2) ⇒ 速度头前最后一次调制把 shift 当 scale 乘、scale 当 shift 加。
    sh, sc = ao.chunk(2, dim=-1)
    _ti = timestep_indices.to(DEV)
    hh = hh * (1.0 + sc.index_select(0, _ti).unsqueeze(0)) + sh.index_select(0, _ti).unsqueeze(0)
    # [FIX-1016 FP32-ISLAND] 调制完升到输出头的 fp32 再进头（官方 660；ComfyUI model.py:320 同）
    hh = hh.to(torch.float32)
    vo = lin(hh, sm["final_layer.video_out.weight"], sm["final_layer.video_out.bias"]).index_select(1, video_indices.to(DEV))
    aout = lin(hh, sm["final_layer.audio_out.weight"], sm["final_layer.audio_out.bias"]).index_select(1, audio_indices.to(DEV))
    return vo, aout


def _vae():
    """加载官方视频 VAE（fp32 CPU→按需 GPU ✅）"""
    import sys as _s
    _s.path.insert(0, H3)
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    print("[v] 准备加载 VAE（legacy native，整份 bf16）…", flush=True)
    import gc as _g3; _g3.collect()
    v = MiniMaxH3VideoVAE.from_pretrained(os.path.join(H3, "video_vae"))
    print("[v] VAE 加载完成 ✅ %.1fM 参数" % (sum(p.numel() for p in v.parameters())/1e6), flush=True)
    _g3.collect(); _g3.collect()
    # [FIX-1017 VAE-BF16] 回退整份 bf16（你确认 bf16 是对的；fp32 整份在 12GB 卡上与 DiT 抢显存，
    #   且同 latent 的 A/B 里 fp32 反而更"软"）。官方 fp32 岛属于 diffusers port 类，而 port 类加载不了
    #   这份 native 权重（strict 缺 544/多 401、无转换代码）⇒ 该岛在此权重路径不可达。
    #   分块保留官方 256/64，可用 H3_VAE_TILE/H3_VAE_DTILE/H3_VAE_OVLP 覆盖；OOM 时 _decode_safe 自动分小。
    v = v.to(dtype=torch.bfloat16)
    v.model.tile_size = int(os.environ.get("H3_VAE_TILE", "256"))
    v.model.decoder_tile_size = int(os.environ.get("H3_VAE_DTILE", "256"))
    v.model.decoder_tile_overlap_min = int(os.environ.get("H3_VAE_OVLP", "64"))
    print("[v] 整份 bf16 ✅ encoder tile=%d decoder tile=%d overlap=%d" % (
        v.model.tile_size, v.model.decoder_tile_size, v.model.decoder_tile_overlap_min), flush=True)
    import json as _j
    c = _j.load(open(os.path.join(H3, "video_vae", "config.json"), encoding="utf-8"))
    class _C:
        def __init__(s2, b, e): s2._b, s2._e = b, e
        def __getattr__(s2, k):
            e = object.__getattribute__(s2, "_e")
            return e[k] if k in e else getattr(object.__getattribute__(s2, "_b"), k)
    v.config = _C(v.config, {k: vv for k, vv in c.items() if not k.startswith("_")})
    import torch.nn as nn
    for m in v.modules():
        if isinstance(m, nn.Conv3d):
            o = m.forward
            m.forward = (lambda oo: (lambda z, *a, **k: oo(z.contiguous(), *a, **k)))(o)
    return v, c

def _decode_safe(v, y):
    """fp32 VAE 解码。12GB 卡上 decoder(ViT, ~9.6GB fp32) 的激活靠分块控：
       OOM 就按"分块分小"逐级降 decoder tile 重试；三级都不行才降 bf16 保出片
       （不让 DiT 那几十分钟的 latent 白费）。latent 的 dtype 对齐 post_quant_conv。"""
    for dt, ov in ((None, None), (96, 24), (64, 16)):
        if dt:
            v.model.decoder_tile_size, v.model.decoder_tile_overlap_min = dt, ov
            print("[v] 解码 OOM ⇒ decoder tile 降到 %d 重试" % dt, flush=True)
        try:
            with torch.no_grad():
                return v.decode_base(y.to(device=DEV, dtype=next(v.model.post_quant_conv.parameters()).dtype))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
    print("[v] fp32 三级分块仍 OOM ⇒ 本次降 bf16 保出片", flush=True)
    v.to(torch.bfloat16)
    with torch.no_grad():
        return v.decode_base(y.to(device=DEV, dtype=next(v.model.post_quant_conv.parameters()).dtype))

def encode_ref(v, c, img_path, dev=DEV):
    from PIL import Image
    import numpy as np
    im = Image.open(img_path).convert("RGB")
    px = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None, :, None].float()   # (1,3,1,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1, 1)
    x = ((px / 255.0) - mean) / std
    # [FIX-1017 ENC-PARTIAL] fp32 VAE 整份 10.4GB，与 DiT 常驻层(5.2GB)同卡必 OOM；
    #   编码只碰 encoder + quant_conv（≈0.75GB），只搬这两个子模块。
    _enc = (v.model.encoder, v.model.quant_conv)
    for m in _enc:
        m.to(dev)
    with torch.no_grad():
        _vdt = next(v.model.encoder.parameters()).dtype
        # [FIX-1016 POST-SAMPLE] 官方 encoders.py:102 encode_vae_condition 三件事旧代码都没做:
        #   ① 单帧走 tiled 的 _adaptive_encode（旧代码用裸 encode()，512 没分块）
        #   ② 后验要 sample(种子42)，不是取 mean —— logvar 那 24 通道不是摆设
        #   ③ 采样后 trim_code 到 1 个 token（causal_encoder 取末尾）
        mom = v._adaptive_encode(x.to(device=dev, dtype=_vdt))
        _mean, _logvar = mom.chunk(2, dim=1)
        _std = torch.exp(0.5 * _logvar.clamp(-30.0, 20.0))
        lat = _mean + _std * torch.randn(_mean.shape, generator=torch.Generator().manual_seed(42),
                                         device="cpu", dtype=_mean.dtype).to(_mean.device)
        lat = v.trim_code(lat, 1)
        lat = lat.to(torch.float16).float()
    lm = torch.tensor(c["latents_mean"]).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)   # ★ 同设备同 dtype ✅
    ls = torch.tensor(c["latents_std"]).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
    lat = (lat - lm) / ls
    for m in _enc:
        m.to("cpu")
    torch.cuda.empty_cache()
    return lat.to(dev)      # (1,24,1,32,32)

def patchify(lat, ps=(1, 2, 2)):
    """(B,C,T,H,W) -> (B, N, C*pt*ph*pw)"""
    B, C, T, Hh, Ww = lat.shape
    pt, ph, pw = ps
    x = lat.reshape(B, C, T // pt, pt, Hh // ph, ph, Ww // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, (T // pt) * (Hh // ph) * (Ww // pw), C * pt * ph * pw)
    return x

def prewarm_all_layers(reader, lf, tag="预读"):
    """注: 40 层超过显存 ⇒ 预读的是"顺读一遍让系统缓存/或进内存"；真正计算时权重在 GPU 槽位 ✅"""
    """★ 一次性顺序读完所有层（顺序 I/O ✅ 加载后不再碰磁盘 ⇒ 系统不卡 ✅ 主人 2026-10-14 口径）"""
    t0 = time.perf_counter()
    try:
        import ctypes
        class _M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        def freemb():
            m = _M(); m.dwLength = ctypes.sizeof(_M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 1048576.0
    except Exception:
        def freemb(): return 99999.0
    for i in range(reader.n):
        lf(i)                       # 触发读入（顺序 ✅ 单线程 ✅）
        if i % 10 == 9:
            print("[e] %s %d/%d  可用内存 %.1fGB  %.0fs" % (tag, i + 1, reader.n, freemb() / 1024, time.perf_counter() - t0), flush=True)
    print("[e] %s 完成 %d 层 %.0fs ✅（之后不再碰磁盘 ✅）" % (tag, reader.n, time.perf_counter() - t0), flush=True)

def run_ref2va(ref, out, steps=4, frames=124, H=512, W=512, prompt="", verbose=True, turbo="none"):
    """参考图 -> 视频（手写全链路 ✅）
       [FIX-1016 TURBO-DEFAULT] 默认不挂蒸馏 LoRA：实测两个 turbo 文件的增量只有权重的
       0.01%~0.04%（ComfyUI 加载器同缩放），对单步速度场近乎惰性；而 4 步 ladder 末段是
       σ 0.8→0 的巨型欧拉步，只有真蒸馏模型才走得动 ⇒ 基座必须给足步数。
       要挂 LoRA 显式传 turbo="4"/"8"。"""
    if turbo != "none":
        load_turbo(TURBO_PATHS[turbo])
    r = Reader(); sm = r.small(); lf = auto_slice(r, resident=3, cache=6, prefetch=True)   # ★ 03:48 出片那次的原配置 ✅
    v, c = _vae()
    import gc as _gc; _gc.collect(); _gc.collect()   # ★ 加载后立刻回收 ✅
    lat = encode_ref(v, c, ref)                      # (1,24,1,H/16,W/16)
    print("[e] 参考图 latent %s ✅" % (tuple(lat.shape),), flush=True)
    xv_ref = patchify(lat)                           # 参考帧 tokens (1, Nr, 96) ✅
    Nr = xv_ref.shape[1]
    # ★ ref2va 条件: 视频序列 = [参考帧(固定) ; 目标帧(去噪)] ✅
    # [FIX-1016 REFCNT] 官方 video_latent_num_frames(num_frames,17,5) 给的是【目标生成帧数】，
    #   参考帧是额外的一行（before_denoise.py:963-967 单独给 reference 加噪到 t=0.999）。
    #   旧代码写 n_lat_total-1，等于少生成一帧，还把参考行当视频内容解码。
    n_lat_total = (frames - 5) // 17 * 5 + 2         # 22 帧 -> 7 个【目标】latent 帧
    # [FIX-1017 CANVAS] 目标 latent 的空间尺寸来自【画布】H/W（官方 before_denoise 的 latent_height/width），
    #   不是参考图：参考按自身分辨率编码（官方 references encoded at their own resolutions），
    #   两者可以不同。旧代码用 lat.shape ⇒ 512 画布+512 参考时巧合相等，768 画布立刻错（1792 vs 4032 行）。
    tgt_lat = torch.randn(1, 24, n_lat_total, H // 16, W // 16, device=lat.device)
    xv_tgt = patchify(tgt_lat)                       # (1, Nt2, 96)
    xv = torch.cat([xv_ref, xv_tgt], dim=1)          # 参考在前 ✅
    Nr_fixed = Nr
    Nv = xv.shape[1]
    # 文本条件: 暂用零条件（编码器接入前 ✅ 先跑通链路）
    _cpt = os.environ.get("H3_COND_PT", "D:/ai/h3_cond.pt")   # [FIX-1015 COND] h3_encode.py 的产物
    cond_tags = None
    if _cpt and os.path.exists(_cpt):
        _cd = torch.load(_cpt, map_location="cpu", weights_only=True)
        _hs = _cd["hidden_states"].float()
        cond = _hs.unsqueeze(0) if _hs.dim() == 2 else _hs
        Nt, txt_dim = cond.shape[1], cond.shape[2]
        # [FIX-1016 VISION-TAG] 官方 before_denoise.py:717-719: prompt_embeds 的标签逐行来自 _build_presentation
        #   —— 视觉块（含 vision_start/vision_end）标 0(video)，真文本行标 1(text)。旧代码把全部条件行标 1，
        #   等于把参考图视觉块当文本喂进 AdaLN 查表。标签由 h3_encode.py 一并算好存盘，这里只做校验。
        _tg = _cd.get("token_tags")
        if _tg is None or _tg.numel() != Nt:
            raise ValueError("cond 文件缺 token_tags（旧版 h3_cond.pt）⇒ 用新版 h3_encode.py 重跑编码")
        cond_tags = _tg.reshape(-1).to(torch.long)
        print("[e] 文本条件已加载: %s <= %s（官方呈现标签, tag=0 行 %d/%d）" % (
            tuple(cond.shape), os.path.basename(_cpt), int((cond_tags == 0).sum()), Nt), flush=True)
    else:
        Nt, txt_dim = 16, 5120
        cond = torch.zeros(1, Nt, txt_dim)
        print("[e] 无 cond 文件, 用零条件: %s" % _cpt, flush=True)
    # [FIX-1016 AUDIO-STREAM] MiniMax-H3 是 joint video+audio 模型（官方只有 t2va/fl2va/ref2va 三个任务，
    #   没有纯视频路径），官方布局 [text | reference | target audio | target video] 里音频行是必需的：
    #   num_audio_latents = round(frames/24*40)，行数 = 该值 × audio_channels(2)，每行 32 维，
    #   初值直接按【行布局】取 randn（官方 before_denoise.py:882-888）。
    #   这里把音频块放在序列末尾而不是官方中间 —— 官方确认无注意力掩码、rope 与 tag 都按行给，
    #   所以行序不影响结果，只影响可读性。
    _AUD_LAT_PER_S, _H3_FPS, _AUD_CH, _AUD_LC = 40, 24, 2, 32
    n_aud = int(round(frames / _H3_FPS * _AUD_LAT_PER_S))
    na_rows = n_aud * _AUD_CH
    xa = torch.randn(1, na_rows, _AUD_LC, device=lat.device)
    vid = torch.arange(0, Nv)
    txt = torch.arange(Nv, Nv + Nt)
    aud = torch.arange(Nv + Nt, Nv + Nt + na_rows)
    L = Nv + Nt + na_rows
    print("[e] 音频流已接入: %d latent × %d 声道 = %d 行 (tag=2, 通道维 %d)" % (
        n_aud, _AUD_CH, na_rows, _AUD_LC), flush=True)
    tags = torch.cat([torch.zeros(Nv, dtype=torch.long),
                      cond_tags if cond_tags is not None else torch.ones(Nt, dtype=torch.long),
                      torch.full((na_rows,), 2, dtype=torch.long)])   # 官方 tag: video=0, text=1, audio=2
    # [FIX-1016 ROPE-POS] 照官方 build_ref2va_packed_sequence 造 (t,h,w) 网格（float64，逐比特对齐）：
    #   ① 文本行占时间轴 0..Nt-1（旧代码全 0，与首帧撞坐标）
    #   ② 图像参考只推进 1 个整数旋转槽（官方注释：一张图 ≠ 一个 latent 帧的 5/3 单位）
    #   ③ 目标帧从参考推进后的时刻起，步长非均匀 5/3*(1,4,4,4,4)（旧代码均匀 +1）
    #   ④ 空间轴用官方居中+宽高比归一网格，方形画布铺 [0,32)（旧代码整数 0..15，尺度差 2 倍）
    pos = torch.zeros(L, 3, dtype=torch.float64)
    pos[Nv:Nv + Nt, 0] = torch.arange(Nt, dtype=torch.float64)
    _pt, _ph, _pw = 1, 2, 2
    rotary_time = float(Nt)
    _ref_h, _ref_w = int(lat.shape[-2]), int(lat.shape[-1])
    _ref_grid, _ = _frame_position_grid(_ref_h, _ref_w, _ph, _pw)
    pos[0:Nr, 0] = rotary_time
    pos[0:Nr, 1:] = _ref_grid
    rotary_time += 1.0
    _tgt_grid, _tgt_wgrid = _frame_position_grid(int(H // 16), int(W // 16), _ph, _pw)
    _rpf = _tgt_grid.shape[0]
    _n_tgt = (Nv - Nr) // _rpf
    pos[Nr:Nv, 0] = _temporal_position_grid(_n_tgt, rotary_time).repeat_interleave(_rpf)
    pos[Nr:Nv, 1:] = _tgt_grid.repeat(_n_tgt, 1)
    # 音频块与目标视频共用同一个 rotary_time（官方 ref2va 就是 refs 推进后的那个时刻）
    _fill_audio_positions(pos, slice(Nv + Nt, L), n_aud, rotary_time, _tgt_wgrid, _AUD_CH)
    print("[e] ROPE-POS 文本 %d 行 / 参考 %d 行(%dx%d) / 目标 %d 帧×%d 行 / 音频 %d 行, 时间起点 %.1f" % (
        Nt, Nr, _ref_h, _ref_w, _n_tgt, _rpf, na_rows, float(Nt) + 1.0), flush=True)
    # [FIX-1016 T-CONV] 官方 H3 约定：t = 1 - sigma 且 **t=1 才是干净**；模型输出是 data-ward velocity，
    #   x0 = x_t + (1-t)*v = x_t + sigma*v；更新 x_next = r*x_t + (1-r)*x0, r = sigma_next/sigma
    #   ⇒ x_next = x_t + (sigma_i - sigma_{i+1})*v。旧代码把 sigma 当 t 传、更新又用减号，两处同时反号。
    # [FIX-1016 COND-T] 条件(参考)行钉在 t = max(t, keyframe_noise_aug=0.999)，且参考 latent 必须先
    #   scale_noise 成 0.999*x0 + 0.001*noise —— 官方注释：模型训练时锚点带极轻噪声，喂 t=1.0 反而出分布。
    KEYFRAME_NOISE_AUG = 0.999
    _g = torch.Generator(device=xv_ref.device).manual_seed(42)          # 官方 keyframe_encode_seed=42
    xv_ref_aug = KEYFRAME_NOISE_AUG * xv_ref + (1.0 - KEYFRAME_NOISE_AUG) * torch.randn(
        xv_ref.shape, generator=_g, device=xv_ref.device, dtype=xv_ref.dtype)
    xt = xv.clone()
    xt[:, :Nr_fixed] = xv_ref_aug
    # [FIX-1015 SIGMA-SHIFT] 官方 MiniMaxH3Scheduler: shift=12.0 的 flow-matching
    # [FIX-1017 GRID] 官方 set_timesteps(n): base=linspace(1,0,n)（**含 0**）→ shift → unique_consecutive，
    #   得 n 个格点、n-1 次评估。旧代码 linspace(1,1/n,n)+补0 = n+1 个格点 ⇒ 比官方多走一个格点
    #   （我的"50 步"实为官方口径 51 格点）。CLI 的 steps 从现在起=官方 num_inference_steps。
    _sh = 12.0
    _u = torch.linspace(1.0, 0.0, steps)
    sigmas = torch.unique_consecutive(_sh * _u / (1 + (_sh - 1) * _u))
    # [FIX-1016 AUDIO-STREAM] 音频用的是【另一个】调度器: audio_scheduler shift=3.0（视频是 12.0），
    #   实测 model_index 目录里 scheduler/scheduler_config.json=12.0、audio_scheduler=3.0。
    _ash = 3.0
    asigmas = torch.unique_consecutive(_ash * _u / (1 + (_ash - 1) * _u))
    _ne = int(sigmas.shape[0]) - 1
    t0 = time.perf_counter()
    for i in range(_ne):
        _t = float(1.0 - sigmas[i])
        _at = float(1.0 - asigmas[i])
        row_t = torch.full((L,), _t, dtype=torch.float32)
        row_t[:Nr_fixed] = max(_t, KEYFRAME_NOISE_AUG)     # 条件(参考)行钉 0.999
        row_t[Nv + Nt:] = _at                              # 生成音频行走音频时刻表
        uniq_t, ti = torch.unique(row_t, sorted=True, return_inverse=True)   # 官方 build_row_timesteps
        vo, ao = dit_forward(r, xt, xa, cond, uniq_t, ti, tags, pos, vid, aud, txt, sm=sm, layer_fn=lf, verbose=verbose)
        xt = xt + (sigmas[i] - sigmas[i + 1]) * vo
        xa = xa + (asigmas[i] - asigmas[i + 1]) * ao       # 音频流同步去噪（joint 模型，不能只更视频）
        xt[:, :Nr_fixed] = xv_ref_aug                    # ★ 每步把(带噪)参考帧替换回去（ref2va 条件 ✅）
        # [FIX-1016 DIAG] 每步打印速度场与潜变量幅度：若 |v| 远小于 latent std，说明网络几乎没在去噪；
        #   若 xt 的目标行 std 一直不降，说明更新方向/尺度有问题。留着比每次靠猜快。
        #   再加一个"每格交替"指标：把目标行还原成 (T,C,H,W) 后求相邻格相关（lag1）。
        #   平滑图 ≈ +0.9、白噪 ≈ 0、2 格周期棋盘 ≈ -0.9 —— 石板有没有被压下去，看这个数最快。
        try:
            _tg = xt[:, Nr_fixed:].float(); _vv = vo.float()
            print("[e] 步%d |v|std=%.4f |v|max=%.3f  目标xt std=%.4f mean=%+.4f  参考行std=%.4f" % (
                i + 1, float(_vv.std()), float(_vv.abs().max()), float(_tg.std()), float(_tg.mean()),
                float(xt[:, :Nr_fixed].float().std())), flush=True)
            _nH = _nW = (H // 16) // 2; _Tt = _tg.shape[1] // (_nH * _nW)
            _vz = vo[:, Nr_fixed:].float().reshape(_Tt, _nH, _nW, 24, 1, 2, 2) \
                     .permute(0, 3, 4, 1, 5, 2, 6).reshape(_Tt, 24, _nH * 2, _nW * 2)
            _l1w = float(torch.corrcoef(torch.stack([_vz[..., :-1].reshape(-1), _vz[..., 1:].reshape(-1)]))[0, 1])
            _l1h = float(torch.corrcoef(torch.stack([_vz[..., :-1, :].reshape(-1), _vz[..., 1:, :].reshape(-1)]))[0, 1])
            print("[e] 步%d v相邻格相关 横=%.3f 纵=%.3f（平滑≈+0.9 白噪≈0 石板≈-0.9）" % (
                i + 1, _l1w, _l1h), flush=True)
        except Exception as _ed:
            print("[e] 诊断打印失败: %s" % str(_ed)[:90], flush=True)
        print("[e] === 采样步 %d/%d 完成 %.0fs | 生成t=%.4f 条件t=%.4f distinct_t=%s ===" % (
            i + 1, _ne, time.perf_counter() - t0, _t, float(row_t[0]),
            [round(float(x), 4) for x in uniq_t]), flush=True)
    # 反 patchify -> latent -> decode
    # [FIX-1016 DROP-REF] 只解目标帧：参考行是条件锚点、不是视频内容。旧代码把它一起解，
    #   于是成片前 4 帧其实是参考图被 VAE 原样 echo —— 看着"与参考相关度 0.985"并不是模型输出。
    _tgt = xt[:, Nr_fixed:]
    B, N2, D = _tgt.shape
    Hh, Ww = H // 16, W // 16
    T = N2 // ((Hh // 2) * (Ww // 2))                # 目标 latent 帧数
    y = _tgt.reshape(B, T, Hh // 2, Ww // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, 24, T, Hh, Ww)
    lm = torch.tensor(c["latents_mean"]).view(1, -1, 1, 1, 1).to(y.device)
    ls = torch.tensor(c["latents_std"]).view(1, -1, 1, 1, 1).to(y.device)
    y = y * ls + lm
    if os.environ.get("H3_SAVE_LAT"):            # ★ 存最终 latent，之后只重跑解码做 A/B ✅
        torch.save(y.cpu(), os.environ["H3_SAVE_LAT"])

    # [FIX-1015 VAE-FREE] 解码前释放 DiT 的 GPU 状态(常驻层+槽位+sm 约7GB)
    # 否则 VAE 只剩 ~5GB 显存 => 贴顶磨 20+ 分钟(03:43-04:04 实测)
    try:
        del lf, r, sm, xt
    except Exception:
        pass
    import gc as _g2
    _g2.collect(); _g2.collect(); torch.cuda.empty_cache()
    print("[e] 解码前释放 DiT 显存, 现空闲 %.2fGB" % ((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**3), flush=True)
    v = v.to(DEV)
    # [FIX-1016 DECODE-TEMPORAL] 官方入口 decode_base => decode_temporal => tiled_decode，
    #   7 latent 正好 17*1+5 = 22 帧 ✅（旧代码调裸 decode() 得 28 帧）；dtype/OOM 由 _decode_safe 管
    dec = _decode_safe(v, y)
    fr = dec[0] if isinstance(dec, (tuple, list)) else dec
    fr = fr[0].permute(1, 2, 3, 0).float().cpu().numpy()
    import numpy as np
    # [FIX-1015 DENORM] encode 用 ImageNet 归一化((px/255-mean)/std) => decode 输出同域(约[-2,2.2])
    # 旧代码直接 clip(0,1) 把负值全削成 0 => 出片全黑(实测 mean=0.027) => 必须先反归一化
    fr = fr * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
    print("[e] 反归一化后帧统计: mean=%.3f std=%.3f min=%.3f max=%.3f" % (fr.mean(), fr.std(), fr.min(), fr.max()), flush=True)
    fr = np.clip(fr, 0, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(out, (fr * 255).astype("uint8"), fps=24)
    print("[e] ★★★ 已写出 " + out + "  帧 %s" % (fr.shape,), flush=True)
    return True

def decode_only(lat_pt, out):
    """只跑 VAE 解码：DiT 的 latent 已缓存，调解码不用重跑十分钟 ✅"""
    v, c = _vae()
    y = torch.load(lat_pt, weights_only=True)
    print("[d] latent %s <= %s" % (tuple(y.shape), lat_pt), flush=True)
    torch.cuda.empty_cache()
    v = v.to(DEV)
    dec = _decode_safe(v, y)
    fr = dec[0].permute(1, 2, 3, 0).float().cpu().numpy()
    import numpy as np
    fr = fr * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array([0.485, 0.456, 0.406], dtype=np.float32)
    print("[d] 反归一化后帧统计: mean=%.3f std=%.3f min=%.3f max=%.3f" % (fr.mean(), fr.std(), fr.min(), fr.max()), flush=True)
    fr = np.clip(fr, 0, 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import imageio.v3 as iio
    iio.imwrite(out, (fr * 255).astype("uint8"), fps=24)
    print("[d] ★★★ 已写出 %s  帧 %s" % (out, fr.shape), flush=True)
    return True

def self_test():
    """随机输入冒烟：验证 50 层能否在 GPU 上跑通并出有限值 ✅（不依赖编码器/VAE ✅）"""
    r = Reader(); sm = r.small(); lf = auto_slice(r, resident=3, cache=6, prefetch=True)   # ★ 03:48 出片那次的原配置 ✅
    Nv, Na, Nt = 64, 8, 12
    L = Nv + Na + Nt
    vid = torch.arange(0, Nv); aud = torch.arange(Nv, Nv + Na); txt = torch.arange(Nv + Na, L)
    xv = torch.randn(1, Nv, 96)
    xa = torch.randn(1, Na, 32)
    cd = torch.randn(1, Nt, 5120)
    ts = torch.tensor([0.7])
    ti = torch.zeros(L, dtype=torch.long)
    tags = torch.cat([torch.zeros(Nv), torch.ones(Nt), torch.full((Na,), 2)]).to(torch.long)
    tags = tags[torch.argsort(torch.cat([vid, txt, aud]))]   # 按打包顺序排列
    pos = torch.zeros(L, 3, dtype=torch.float64)   # 与主路径同 dtype（旧为 long，会把非整数旋转坐标截断）
    t0 = time.perf_counter()
    vo, ao = dit_forward(r, xv, xa, cd, ts, ti, tags, pos, vid, aud, txt, sm=sm, layer_fn=lf, verbose=True)
    print("[e] ★ 前向 OK %.1fs  video=%s audio=%s  finite=%s/%s" % (
          time.perf_counter()-t0, tuple(vo.shape), tuple(ao.shape), bool(torch.isfinite(vo).all()), bool(torch.isfinite(ao).all())), flush=True)
    print("[e] video 统计: min=%.3f max=%.3f mean=%.4f" % (vo.min().item(), vo.max().item(), vo.mean().item()), flush=True)
    return True

if __name__ == "__main__":
    r = Reader()
    import sys
    if "--run" in sys.argv:
        import time
        _i = sys.argv.index("--run")
        run_ref2va(sys.argv[_i + 1], sys.argv[_i + 2], steps=int(sys.argv[_i + 3]),
                   frames=int(os.environ.get("H3_FRAMES", "124")),
                   H=int(os.environ.get("H3_H", "512")), W=int(os.environ.get("H3_W", "512")),
                    turbo=(sys.argv[_i + 4] if len(sys.argv) > _i + 4 else "none"))
        os._exit(0)
    if "--decode" in sys.argv:
        _i = sys.argv.index("--decode")
        decode_only(sys.argv[_i + 1], sys.argv[_i + 2])
        os._exit(0)
    if "--selftest" in sys.argv:
        import time
        ok = self_test()
        import gc; gc.collect(); torch.cuda.empty_cache()
        os._exit(0 if ok else 1)
    print("层数 %d · 非层权重 %d ✅" % (r.n, len(r.other)), flush=True)
    sm = r.small(dev="cpu")
    for k in sorted(sm)[:12]: print("   ", k, tuple(sm[k].shape), flush=True)
    for i in (0, r.n - 1):
        st = r.layer(i, dev="cpu")
        print("层 %d: %d 个张量 %s ✅" % (i, len(st), sorted(st)[:4]), flush=True)
    os._exit(0)