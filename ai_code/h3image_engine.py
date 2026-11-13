# -*- coding: utf-8 -*-
"""MiniMax-H3-Image 本地流式推理引擎（12GB 卡 / 512x512 冒烟）

路线 C：不装 diffusers main、不做 131G 权重转换，直读原生分片。
  · DiT：复用 D:\\ai\\h3_engine.py 已验证的逐层 pread + 手写算子（50 层 / 5376 / 56 头）；
         改为 2 模态布局 [text | generated image]、无音频流、PDD 6 步每步一个输出头。
  · 文本条件：复用盘上 h3_cond_*.pt（Qwen3-VL 第 50 层 hidden，63G 编码器不进内存）。
  · 图像 VAE：按 diffusers main 的 AutoencoderKLMiniMaxH3Image / MiniMaxH3ImageViTDecoder2d
         逐行复刻，直读 vae/diffusion_pytorch_model.safetensors 原生权重。

官方口径来源（2026-11-11 从 diffusers main 取回，存于 D:\\ai\\cache\\temp\\h3image_src\\）：
  modular_pipelines/minimax_h3_image/{before_denoise,denoise,decoders,modular_pipeline}.py
  models/autoencoders/autoencoder_kl_minimax_h3_image.py
  models/transformers/transformer_minimax_h3_image.py
  schedulers/scheduling_minimax_h3.py

★ 画布/文本编码口径按 diffusers main 逐项对齐（2026-11-11 改）：
  · get_image_canvas_size / min_generation_area 默认 2048²，输出**原生生成画布，不做任何缩小**
  · 文本条件改走官方 t2i 呈现（h3image_encode.py），不再复用视频任务的 cond
  · 噪声按官方 PrepareLatentsStep + patchify_video_latents 生成

用法:
  # 0. 先生成官方 t2i 文本条件（独立子进程，63G 编码器走磁盘 offload）
  python h3image_encode.py --prompt "<详细英文描述>" --out D:\\ai\\h3image_cond_t2i.pt
  # 1. 只看官方画布/序列计划（不加载模型）
  python h3image_engine.py --plan --size 512
  # 2. 完整生成，默认 2048x2048 原生画布直出
  python h3image_engine.py --run --seed 42
  # 3. 16MP 官方最佳工作点
  python h3image_engine.py --run --width 4096 --height 4096 --min_area 16777216
  # 4. 只按请求尺寸生成（等价于官方传 min_generation_area=1024）
  python h3image_engine.py --run --size 512 --min_area 1024
  python h3image_engine.py --vae_selftest
  python h3image_engine.py --decode_only <latent.pt> --out <png>
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("H3_LOG", r"D:\ai\h3image_run.log")

import torch
import torch.nn.functional as F

ROOT = r"D:\ai\models\MiniMax-H3-Image"
TF_DIR = os.path.join(ROOT, "transformer")
VAE_FILE = os.path.join(ROOT, "vae", "diffusion_pytorch_model.safetensors")
VAE_CFG = os.path.join(ROOT, "vae", "config.json")
PDD_FILE = os.path.join(ROOT, "pdd_fused_heads.safetensors")
PDD_CFG = os.path.join(ROOT, "pdd_config.json")
ENGINE = r"D:\ai\h3_engine.py"
DEFAULT_COND = r"D:\ai\h3image_cond_t2i.pt"   # 官方 t2i 呈现，由 h3image_encode.py 产出

DEV = "cuda"
DTYPE = torch.bfloat16
HID = 5376
HEADS, HEAD_DIM = 56, 128
MODE_NUM = 3              # AdaLN 表 96768 = 6*5376*3，去音频但表仍是 3 模态
LATENT_CH = 24
PATCH = (1, 2, 2)
VAE_RATIO = 16
IMAGE_TAG, TEXT_TAG = 0, 1
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# --- 官方 modular_pipelines/minimax_h3_image 常量（逐项对照，勿随手改） ---
CANVAS_MULTIPLE = 32                    # vae_spatial_compression_ratio(16) * patch_w(2)
MIN_GENERATION_AREA = 2048 * 2048        # PrepareCanvasStep 默认 4194304（4MP 有效区间下限）
MIN_AREA_FLOOR = 1024                   # 官方下限；传 1024 = 永远按请求尺寸生成
MIN_AREA_CEIL = 4096 * 4096             # 官方上限，16MP 为官方最佳工作点
MAX_SEQUENCE_LENGTH = 4096              # 官方提示词 token 上限
TEXT_ENCODER_LAYER = 50                 # 官方 hidden_states[50]
TILE_SIZE = 256                         # AutoencoderKLMiniMaxH3Image.tile_sample_min_*
TILE_OVERLAP_MIN = 64                   # AutoencoderKLMiniMaxH3Image.tile_*_overlap_*

_LOG = r"D:\ai\h3image_engine.log"
_fh = None


def log(*a):
    global _fh
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        if _fh is None:
            _fh = open(_LOG, "a", encoding="utf-8", buffering=1)
        _fh.write(s + "\n")
    except Exception:
        pass


def load_engine():
    """加载 h3_engine 的算子并把权重根改指 H3-Image。"""
    spec = importlib.util.spec_from_file_location("h3_engine_lib", ENGINE)
    he = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(he)
    he.TF = TF_DIR
    he.H3 = ROOT
    return he


def get_image_canvas_size(height, width, min_generation_area, multiple=CANVAS_MULTIPLE):
    """官方 before_denoise.py:94-108 get_image_canvas_size，逐行照抄。

    小画布按面积等比放大（保持宽高比），两侧 round 到 32 的倍数；
    round 用 Python 内建（banker's rounding），不能换成 int()/floor。
    """
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def resolve_canvas(args):
    """官方 MiniMaxH3ImagePrepareCanvasStep 的校验 + 画布解析。

    ★ 按主人要求：不设置 output_height / output_width，即跳过官方 DecodeStep 里
      「把大画布 resize 回请求尺寸」那一步 —— 输出原生生成画布，质量上不做任何缩小。
      （官方 DecodeStep 在 output_* 为 None 时本来就用 image.shape，属官方口径内。）
    """
    if not MIN_AREA_FLOOR <= args.min_area <= MIN_AREA_CEIL:
        raise ValueError("min_area 必须在 [%d, %d] 内，得到 %d" % (
            MIN_AREA_FLOOR, MIN_AREA_CEIL, args.min_area))
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height/width 必须为正，得到 %dx%d" % (args.height, args.width))
    H, W = get_image_canvas_size(args.height, args.width, args.min_area)
    if H % CANVAS_MULTIPLE or W % CANVAS_MULTIPLE:
        raise ValueError("画布必须是 %d 的倍数，得到 %dx%d" % (CANVAS_MULTIPLE, H, W))
    return H, W


def plan_canvas(args):
    """只打印官方口径的画布/序列/VAE 分块计划，不加载任何模型。"""
    H, W = resolve_canvas(args)
    lh, lw = H // VAE_RATIO, W // VAE_RATIO
    rows = (lh // PATCH[1]) * (lw // PATCH[2])
    yp, _, _ = _split_tiles(H, TILE_SIZE, TILE_OVERLAP_MIN)
    xp, _, _ = _split_tiles(W, TILE_SIZE, TILE_OVERLAP_MIN)
    log("[plan] 请求 %dx%d, min_area=%d -> 生成画布 %dx%d (%.2f MP)" % (
        args.height, args.width, args.min_area, H, W, H * W / 1e6))
    log("[plan] latent %dx%d -> %d 图像行 + 文本行; VAE tiling %dx%d 个 tile" % (
        lh, lw, rows, len(yp), len(xp)))
    log("[plan] 输出 = 原生画布 %dx%d，不做任何 resize / 下采样" % (H, W))
    log("[plan] 官方有效区间 4-20MP，16MP 最佳；min_area 上限 %d" % MIN_AREA_CEIL)
    # 显存/耗时粗估（512 实测：2.24s/层，其中磁盘搬运约 1.6s、GPU 计算约 0.64s）
    L = rows + 300
    per_row = (5376 + 21504 + 28672 + 7168 + 6 * 5376 + 5376) * 2   # bf16 上界
    act_gb = L * per_row / 1024 ** 3
    t_layer = 1.6 + 0.64 * L / 283.0
    log("[plan] 粗估: 序列 ~%d 行, 激活 %.2f GB + 权重 ~6.45 GB(常驻%d层) = %.2f GB / 11.91 GB" % (
        L, act_gb, args.resident, act_gb + 6.45))
    log("[plan] 粗估: 6步x50层 约 %.0f 分钟（线性外推，注意力 L² 会再增加）" % (
        6 * 50 * t_layer / 60))
    if act_gb + 6.45 > 10.5:
        log("[plan][warn] 显存偏紧，建议加 --resident 2（少常驻一层省约 1.26GB）")
    for s in (512, 1024, 2048, 4096):
        h, w = get_image_canvas_size(s, s, args.min_area)
        n = (h // 32) ** 2
        log("[plan]   请求 %4d -> 画布 %4dx%-4d  %6d 图像行 (%5.2f MP)" % (
            s, h, w, n, h * w / 1e6))
    return True


def build_layout(he, n_text, latent_h, latent_w):
    """官方 build_image_packed_sequence 的 text2image 分支：[text | generated image]。"""
    _, ph, pw = PATCH
    n_img = (latent_h // ph) * (latent_w // pw)
    L = n_text + n_img
    text_idx = torch.arange(n_text)
    image_idx = torch.arange(n_text, L)
    tags = torch.empty(L, dtype=torch.long)
    tags[text_idx] = TEXT_TAG
    tags[image_idx] = IMAGE_TAG

    pos = torch.zeros(L, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    grid, _ = he._frame_position_grid(latent_h, latent_w, ph, pw)
    pos[n_text:, 0] = float(n_text)
    pos[n_text:, 1:] = grid
    return pos, tags, image_idx, text_idx, n_img


def dit_forward_image(he, reader, ximg, cond, uniq_t, ti, tags, pos,
                      image_idx, text_idx, sm, layer_fn, pdd_w, pdd_b):
    """一次 H3-Image DiT 前向：官方 MiniMaxH3ImageTransformer2DModel.forward 的逐层流式版。"""
    t0 = time.perf_counter()
    # 1. 各模态投影 + 打包（text 在前，image 在后）
    ve = he.lin(ximg.to(DEV, sm["video_patch_proj.weight"].dtype),
                sm["video_patch_proj.weight"], sm["video_patch_proj.bias"])
    te = he.lin(cond.to(DEV, DTYPE), sm["condition_proj.weight"], sm["condition_proj.bias"])
    te = he.refiner(te, None, sm, he.refiner_lora())
    L = pos.shape[0]
    h = te.new_zeros((te.shape[0], L, te.shape[-1]))
    h = h.index_copy(1, text_idx.to(DEV), te)
    h = h.index_copy(1, image_idx.to(DEV), ve.to(te.dtype))

    # 2. timestep 嵌入（t = 1 - sigma，t=1 为干净）
    temb = he.time_embed(uniq_t.to(DEV))
    temb = he.lin(temb, sm["time_embedder.proj_in.weight"], sm["time_embedder.proj_in.bias"])
    temb = he.lin(F.silu(temb), sm["time_embedder.proj_out.weight"], sm["time_embedder.proj_out.bias"])

    # 3. AdaLN 行索引 = timestep_indices * 模态数 + token_tags
    idx = (ti.to(DEV) * MODE_NUM + tags.to(DEV)).long()
    cos, sin = he.build_rope(pos.to(DEV), sm["rope.inv_freq"].float())

    for i in range(reader.n):
        st = layer_fn(i)
        aw = st["adaln_proj.linear.weight"]
        a = he.lin(F.silu(temb.float()).to(aw.dtype), aw,
                   st["adaln_proj.linear.bias"]).view(-1, 6 * HID)
        mods = a.chunk(6, dim=-1)
        h = he.block_forward(h, st, mods, idx, cos, sin, lora=he.lora_side(i))
        if i % 10 == 0 or i == reader.n - 1:
            log("[d] layer %d/%d  %.1fs  显存 %.2fGB" % (
                i, reader.n, time.perf_counter() - t0,
                torch.cuda.memory_allocated() / 1024 ** 3))

    # 4. 末层 AdaLayerNormOut + 本步专用 PDD 输出头
    hh = he.rms(h, sm["final_layer.norm.weight"])
    fw = sm["final_layer.adaln_proj.linear.weight"]
    ao = he.lin(F.silu(temb.float()).to(fw.dtype), fw,
                sm["final_layer.adaln_proj.linear.bias"]).view(-1, 2 * HID)
    sh, sc = ao.chunk(2, dim=-1)
    _ti = ti.to(DEV)
    hh = hh * (1.0 + sc.index_select(0, _ti).unsqueeze(0)) + sh.index_select(0, _ti).unsqueeze(0)
    hh = hh.to(torch.float32)
    out = he.lin(hh, pdd_w, pdd_b).index_select(1, image_idx.to(DEV))
    return out


def load_vae():
    from safetensors.torch import load_file
    sd = load_file(VAE_FILE)
    cfg = json.load(open(VAE_CFG, encoding="utf-8"))
    sd = {k: v.to(DEV) for k, v in sd.items()}
    return sd, cfg


def _vae_decode_core(z, sd, cfg, opt=None):
    """单个 latent tile 的解码核心：MiniMaxH3ImageViTDecoder2d.forward。

    opt 用于口径 A/B（默认即正确口径）：
      qkv   : interleaved | stacked —— ★原生 vae 的 to_qkv 是逐头 [q k v] 交错，
               diffusers port 加载时才 de-interleave；直读原生必须按 interleaved 拆，
               否则相邻 latent token 相位错开 ⇒ 16 像素周期网格（2026-11-11 实测 grid 2.90→0.91）。
      ffn   : native | port          —— SwiGLU 两半顺序
      layout: cphw | phwc | cwhp | hwcp —— proj_out 768 维的像素块排列（cphw 为官方）
    """
    opt = opt or {}
    B, C, Hh, Ww = z.shape
    n_tok = Hh * Ww
    x = F.conv2d(z.float(), sd["post_quant_conv.weight"], sd["post_quant_conv.bias"])
    x = x.permute(0, 2, 3, 1).reshape(B, n_tok, C)
    x = F.linear(x, sd["decoder.x_embedder.weight"], sd["decoder.x_embedder.bias"])

    reg = sd["decoder.register_tokens"].expand(B, -1, -1).to(x.dtype)
    cls = torch.zeros((B, 1, x.shape[-1]), device=x.device, dtype=x.dtype)
    x = torch.cat([x, reg, cls], dim=1)
    L = x.shape[1]

    grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=x.device) / size) - 1.0
             for size in (Hh, Ww)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
    pos = torch.cat([pos, pos.new_zeros((reg.shape[1] + 1, 2))], dim=0).unsqueeze(0)

    # 2D RoPE: dim=32, num_axes=2, theta=100
    inv_freq = 1.0 / 100.0 ** torch.arange(0, 1, 2 * 2 / 32, dtype=torch.float32, device=x.device)
    angles = 2.0 * math.pi * pos[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).tile(2).unsqueeze(2)      # (B, L, 1, 32)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

    n_layers = cfg["vit_decoder_kwargs"]["num_layers"]
    heads = cfg["vit_decoder_kwargs"]["heads"]
    dh = cfg["vit_decoder_kwargs"]["dim_head"]
    eps = 1e-5
    for i in range(n_layers):
        p = "decoder.transformer_blocks.%d." % i
        n1 = F.rms_norm(x.float(), (1024,), sd[p + "norm1.weight"].float(), eps=eps).to(x.dtype)
        qkv = F.linear(n1, sd[p + "attn.to_qkv.weight"], sd[p + "attn.to_qkv.bias"])
        if opt.get("qkv", "interleaved") == "interleaved":
            t = qkv.view(B, L, heads, 3, dh)
            q, k, v = (t[..., j, :].reshape(B, L, heads, dh) for j in range(3))
        else:
            q, k, v = (t.view(B, L, heads, dh) for t in qkv.chunk(3, dim=-1))
        q = F.rms_norm(q.float(), (dh,), None, eps=eps).to(q.dtype)
        k = F.rms_norm(k.float(), (dh,), None, eps=eps).to(k.dtype)
        rd = cos.shape[-1]
        qr, qp = q[..., :rd], q[..., rd:]
        q1, q2 = qr.chunk(2, dim=-1)
        q = torch.cat([qr * cos + torch.cat([-q2, q1], dim=-1) * sin, qp], dim=-1)
        kr, kp = k[..., :rd], k[..., rd:]
        k1, k2 = kr.chunk(2, dim=-1)
        k = torch.cat([kr * cos + torch.cat([-k2, k1], dim=-1) * sin, kp], dim=-1)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        o = o.transpose(1, 2).reshape(B, L, heads * dh)
        o = F.linear(o, sd[p + "attn.to_out.weight"], sd[p + "attn.to_out.bias"])
        x = x + o * sd[p + "scale1"]
        n2 = F.rms_norm(x.float(), (1024,), sd[p + "norm2.weight"].float(), eps=eps).to(x.dtype)
        a, g = F.linear(n2, sd[p + "ff.w1.weight"], sd[p + "ff.w1.bias"]).chunk(2, dim=-1)
        ffn = a * F.silu(g) if opt.get("ffn") == "port" else F.silu(a) * g
        o2 = F.linear(ffn, sd[p + "ff.w2.weight"], sd[p + "ff.w2.bias"])
        x = x + o2 * sd[p + "scale2"]

    x = F.layer_norm(x.float(), (1024,), sd["decoder.norm_out.weight"],
                     sd["decoder.norm_out.bias"], eps=eps)
    x = F.linear(x.to(sd["decoder.proj_out.weight"].dtype),
                 sd["decoder.proj_out.weight"], sd["decoder.proj_out.bias"])
    x = x[:, :n_tok, :]
    layout = opt.get("layout", "cphw")
    if layout == "cphw":
        x = x.view(B, Hh, Ww, 3, 16, 16).permute(0, 3, 1, 4, 2, 5)
    elif layout == "phwc":
        x = x.view(B, Hh, Ww, 16, 16, 3).permute(0, 5, 1, 3, 2, 4)
    elif layout == "cwhp":
        x = x.view(B, Hh, Ww, 3, 16, 16).permute(0, 3, 1, 5, 2, 4)
    elif layout == "hwcp":
        x = x.view(B, Hh, Ww, 16, 16, 3).permute(0, 5, 1, 4, 2, 3)
    else:
        raise ValueError("unknown layout " + layout)
    return x.contiguous().reshape(B, 3, Hh * 16, Ww * 16)


def _split_tiles(length, tile_size, min_overlap, ratio=VAE_RATIO):
    """官方 AutoencoderKLMiniMaxH3Image._split_tiles（像素空间切块，重叠按 16 的倍数加宽）。"""
    if tile_size >= length:
        return [0], [length], []
    n = math.ceil(length / tile_size)
    while tile_size * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile_size * n - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * n, overlaps


def _blend(a, b, extent, dim):
    """官方 _blend：a 尾与 b 头线性交叉淡化。"""
    extent = min(a.shape[dim], b.shape[dim], extent)
    pos = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    wa = (1 - pos / extent).view(shape)
    wb = (pos / extent).view(shape)
    sa = [slice(None)] * a.ndim
    sa[dim] = slice(-extent, None)
    sb = [slice(None)] * b.ndim
    sb[dim] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[dim]:
        return blended
    sr = [slice(None)] * b.ndim
    sr[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(sr)]], dim=dim)


def _stitch_tiles(tiles, y_overlaps, x_overlaps):
    """官方 _stitch_tiles。"""
    result_rows = []
    for i, row in enumerate(tiles):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(tiles[i - 1][j], tile, y_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = _blend(row[j - 1], tile, x_overlaps[j - 1], dim=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -y_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -x_overlaps[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def vae_decode(z, sd, cfg, opt=None):
    """官方 AutoencoderKLMiniMaxH3Image.decode：use_tiling=True 为默认。

    512x512 -> 3x3 个 256px tile，重叠 128px，线性混合后拼回。
    官方注释明确 "the released images are the blended-tile ones, so disabling tiling changes the output"。
    opt['tiling']=False 可退回整块解码做 A/B。
    """
    opt = opt or {}
    if opt.get("tiling") is False:
        return _vae_decode_core(z, sd, cfg, opt)
    height = z.shape[-2] * VAE_RATIO
    width = z.shape[-1] * VAE_RATIO
    y_pos, y_len, y_ov = _split_tiles(height, TILE_SIZE, TILE_OVERLAP_MIN)
    x_pos, x_len, x_ov = _split_tiles(width, TILE_SIZE, TILE_OVERLAP_MIN)
    log("[v] tiling: %dx%d 个 tile, y重叠=%s x重叠=%s" % (
        len(y_pos), len(x_pos), y_ov, x_ov))
    rows = []
    for ip, il in zip(y_pos, y_len):
        row = []
        for jp, jl in zip(x_pos, x_len):
            tile = z[..., ip // VAE_RATIO: ip // VAE_RATIO + il // VAE_RATIO,
                      jp // VAE_RATIO: jp // VAE_RATIO + jl // VAE_RATIO]
            row.append(_vae_decode_core(tile, sd, cfg, opt))
        rows.append(row)
    return _stitch_tiles(rows, y_ov, x_ov)


def to_png(image, out):
    from PIL import Image
    import numpy as np
    pm = torch.tensor(PIXEL_MEAN, device=image.device).view(1, -1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=image.device).view(1, -1, 1, 1)
    img = (image.float() * ps + pm).clamp(0, 1)
    img = (img * 255).clamp(0, 255).to(torch.uint8).float() / 255.0   # 官方 8bit 截断量化
    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(arr).save(out)
    log("[v] 已写出 %s  shape=%s  std=%.4f mean=%.4f" % (
        out, arr.shape, float(arr.std()), float(arr.mean())))
    return out


def load_cond(path, mode):
    """官方 t2i 条件：tokenizer(prompt, add_special_tokens=False)[:4096] 的 hidden_states[50]，
    每一行 tag=1（encoders.py:156-168）。由 h3image_encode.py 产出。

    mode=official  : 不做任何过滤，整段提示词全用，tags 由 build_layout 置全 1（官方口径）。
    mode=text_only : 兼容旧视频任务 cond，只保留 tag=1 行（冒烟用，非官方 t2i 呈现）。
    mode=full      : 原样保留，不做过滤也不做校验。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "缺官方 t2i 文本条件 %s\n"
            "  请先运行: D:\\ai\\python\\python.exe D:\\ai\\h3image_encode.py "
            "--prompt \"<详细英文描述>\" --out %s\n"
            "  （仅做链路冒烟可显式 --cond D:\\ai\\h3_cond_apple.pt --cond_mode text_only）" % (path, path))
    d = torch.load(path, map_location="cpu", weights_only=True)
    hs = d["hidden_states"]
    hs = hs.unsqueeze(0) if hs.dim() == 2 else hs
    tags = d.get("token_tags")
    if mode == "text_only" and tags is not None:
        keep = (tags.reshape(-1) == TEXT_TAG)
        if keep.sum() > 0:
            hs = hs[:, keep]
    elif mode == "official":
        if hs.shape[1] > MAX_SEQUENCE_LENGTH:
            hs = hs[:, :MAX_SEQUENCE_LENGTH]
        if tags is not None and int((tags.reshape(-1) != TEXT_TAG).sum()):
            log("[c][warn] cond 含非文本 tag —— 官方 t2i 呈现应全为 1；"
                "建议用 h3image_encode.py 重新编码")
    log("[c] 条件 %s  hidden=%s  dtype=%s  mode=%s  prompt=%r" % (
        os.path.basename(path), tuple(hs.shape), hs.dtype, mode,
        str(d.get("prompt", ""))[:70]))
    return hs.to(torch.bfloat16)


def run(args):
    from safetensors.torch import load_file
    he = load_engine()
    t_all = time.perf_counter()

    H, W = resolve_canvas(args)
    lh, lw = H // VAE_RATIO, W // VAE_RATIO
    if lh % PATCH[1] or lw % PATCH[2]:
        raise ValueError("latent 必须能被 2x2 patch 整除")
    log("[p] 官方画布: 请求 %dx%d / min_area=%d -> 生成 %dx%d (%.2f MP)；"
        "输出原生画布，不做任何缩小" % (
            args.height, args.width, args.min_area, H, W, H * W / 1e6))

    cond = load_cond(args.cond, args.cond_mode)
    n_text = cond.shape[1]
    pos, tags, image_idx, text_idx, n_img = build_layout(he, n_text, lh, lw)
    log("[p] 画布 %dx%d -> latent %dx%d -> %d 图像行 + %d 文本行 = %d 行" % (
        H, W, lh, lw, n_img, n_text, n_img + n_text))

    pdd = load_file(PDD_FILE)
    pdd_w = pdd["video_out.weight"].to(DEV).float()      # (6,96,5376)
    pdd_b = pdd["video_out.bias"].to(DEV).float()        # (6,96)
    cfg_pdd = json.load(open(PDD_CFG, encoding="utf-8"))
    sig0 = torch.tensor(cfg_pdd["sigmas"], dtype=torch.float64)
    # 官方 SetTimestepsStep：把 sigma 先经 float32 的 t=1-sigma 往返一次
    ts = torch.tensor([1.0 - float(s) for s in sig0[:-1]], dtype=torch.float32)
    sigmas = torch.cat([1.0 - ts, torch.zeros(1)]).to(torch.float64)
    log("[s] 6 步 sigma=%s" % [round(float(s), 6) for s in sigmas])

    # 官方 PrepareLatentsStep (before_denoise.py:480-495)：
    #   randn_tensor((1,24,1,lh,lw), cpu generator, float32) + patchify_video_latents((1,2,2))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    lat0 = torch.randn((1, LATENT_CH, 1, lh, lw), generator=g, dtype=torch.float32)
    x = lat0.reshape(1, LATENT_CH, 1, 1, lh // 2, 2, lw // 2, 2) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape(1, n_img, LATENT_CH * PATCH[1] * PATCH[2]).contiguous().to(DEV)

    reader = he.Reader()
    sm = reader.small()
    layer_fn = he.auto_slice(reader, resident=args.resident, cache=args.cache, prefetch=True)
    log("[m] 权重 %d 层 / 小件 %d 项，常驻 GPU %d 层；显存 %.2fGB" % (
        reader.n, len(reader.other), args.resident,
        torch.cuda.memory_allocated() / 1024 ** 3))

    for i in range(int(cfg_pdd["num_inference_steps"])):
        t_i = float(ts[i])
        row_t = torch.full((n_img + n_text,), t_i, dtype=torch.float32)
        uniq_t, ti = torch.unique(row_t, sorted=True, return_inverse=True)
        v = dit_forward_image(he, reader, x, cond, uniq_t, ti, tags, pos,
                              image_idx, text_idx, sm, layer_fn,
                              pdd_w[i], pdd_b[i])
        ds = float(sigmas[i] - sigmas[i + 1])
        x = x + ds * v.float()
        log("[s] 步%d/%d t=%.6f ds=%.6f |v|std=%.4f |v|max=%.3f x_std=%.4f x_mean=%+.4f  %.0fs" % (
            i + 1, int(cfg_pdd["num_inference_steps"]), t_i, ds,
            float(v.float().std()), float(v.float().abs().max()),
            float(x.std()), float(x.mean()), time.perf_counter() - t_all))
        if not bool(torch.isfinite(x).all()):
            log("[!] 出现非有限值，中止")
            return False

    # 反 patchify -> (1,24,lh,lw)
    rows = x[0].float().cpu()
    y = rows.reshape(-1, lh // 2, lw // 2, LATENT_CH, 2, 2) \
            .permute(0, 3, 1, 4, 2, 5).reshape(1, LATENT_CH, lh, lw).contiguous()
    lm = torch.tensor(json.load(open(VAE_CFG, encoding="utf-8"))["latents_mean"]).view(1, -1, 1, 1)
    ls = torch.tensor(json.load(open(VAE_CFG, encoding="utf-8"))["latents_std"]).view(1, -1, 1, 1)
    y = y * ls + lm
    lat_path = args.save_latent or os.path.join(
        r"D:\ai\output", "h3image_%dx%d_latent.pt" % (H, W))
    torch.save(y, lat_path)
    log("[d] latent %s -> %s；释放 DiT 显存" % (tuple(y.shape), lat_path))

    del layer_fn, reader, sm, x, v
    import gc
    gc.collect(); gc.collect(); torch.cuda.empty_cache()
    log("[d] 解码前空闲 %.2fGB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))

    sd, vcfg = load_vae()
    with torch.no_grad():
        image = vae_decode(y.to(DEV), sd, vcfg)
    out = args.out or os.path.join(
        r"D:\ai\output\images", "h3image_%dx%d_%d.png" % (H, W, args.seed))
    to_png(image, out)          # ★ 原生画布直出，绝不 resize
    log("[done] 全流程 %.0fs" % (time.perf_counter() - t_all))
    return True


def vae_selftest(args):
    sd, cfg = load_vae()
    s = args.size or 512
    y = torch.randn(1, LATENT_CH, s // 16, s // 16, device=DEV) * 0.5
    lm = torch.tensor(cfg["latents_mean"], device=DEV).view(1, -1, 1, 1)
    ls = torch.tensor(cfg["latents_std"], device=DEV).view(1, -1, 1, 1)
    with torch.no_grad():
        im = vae_decode(y * ls + lm, sd, cfg)
    log("[selftest] VAE 输出 %s finite=%s" % (tuple(im.shape), bool(torch.isfinite(im).all())))
    to_png(im, args.out or r"D:\ai\output\images\h3image_vae_selftest.png")
    return True


def decode_only(args):
    y = torch.load(args.decode_only, map_location="cpu", weights_only=True)
    sd, cfg = load_vae()
    with torch.no_grad():
        im = vae_decode(y.to(DEV), sd, cfg)
    to_png(im, args.out or r"D:\ai\output\images\h3image_decode.png")
    return True


def main():
    ap = argparse.ArgumentParser(description="MiniMax-H3-Image 流式推理（画布/编码口径逐项对齐 diffusers main）")
    ap.add_argument("--run", action="store_true", help="完整 6 步生成")
    ap.add_argument("--plan", action="store_true", help="只打印官方画布/序列计划，不加载任何模型")
    ap.add_argument("--vae_selftest", action="store_true")
    ap.add_argument("--decode_only", default=None)
    ap.add_argument("--width", type=int, default=2048, help="官方默认 2048")
    ap.add_argument("--height", type=int, default=2048, help="官方默认 2048")
    ap.add_argument("--size", type=int, default=None, help="正方形快捷，覆盖 width/height")
    ap.add_argument("--min_area", type=int, default=MIN_GENERATION_AREA,
                    help="官方最小生成面积，默认 2048²；1024=按请求尺寸原样生成；上限 4096²(16MP 最佳)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond", default=DEFAULT_COND, help="官方 t2i 条件（h3image_encode.py 产出）")
    ap.add_argument("--cond_mode", choices=["official", "text_only", "full"], default="official")
    ap.add_argument("--resident", type=int, default=3)
    ap.add_argument("--cache", type=int, default=4)
    ap.add_argument("--out", default=None, help="默认按画布尺寸命名，原生画布直出、不缩小")
    ap.add_argument("--save_latent", default=None)
    a = ap.parse_args()
    if a.size is not None:
        a.width = a.height = a.size
    log("=" * 70)
    log("[h3image] start %s  width=%d height=%d min_area=%d seed=%d" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), a.width, a.height, a.min_area, a.seed))
    if a.plan:
        ok = plan_canvas(a)
    elif a.vae_selftest:
        ok = vae_selftest(a)
    elif a.decode_only:
        ok = decode_only(a)
    elif a.run:
        ok = run(a)
    else:
        ap.print_help(); ok = False
    log("[h3image] RESULT: %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
