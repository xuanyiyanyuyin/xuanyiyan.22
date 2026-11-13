# -*- coding: utf-8 -*-
"""H3-Image 三层流式引擎（2026-11-11 建，只写未跑，点火需主人下令）

口径移植自 h3_3tier_v2.py（pread 真权重 / CPU LRU 层缓存 / GPU 常驻层 /
单槽位流式层 / p.data 指针换入换出 / 预取线程 / 每层打印），两处适配：
  ① 官方 ModularPipeline 编排**原样保留**（打包/rope/调度/PDD 头选择/VAE 全走官方块），
     流式化只包在两个大模块的**逐层 forward** 上 ⇒ 不改任何官方前向逻辑；
  ② 权重读**转换后 diffusers 布局**（键名=模块路径，恒等映射，无别名/qkv 交织风险；
     交织 qkv 与 SwiGLU 门序由官方转换脚本处理，见 convert_minimax_h3_to_diffusers.py:131）。

前置（都未执行）：
  1. 隔离环境 D:\\ai\\python_h3image（已建：diffusers 0.42.0.dev0 + 基座 torch 2.11 cu128）
  2. 权重转换 D:\\ai\\h3image_convert.py → D:\\ai\\models\\MiniMax-H3-Image-diffusers
     （transformer_diffusers/ + vae_diffusers/；编码器不转，直接流式读本地 H3 原生权重）

显存账（5070 12GB，预注册未实测）：
  小件常驻 ≈ 3GB（token_refiner 2层≈2.5GB + fp32 proj_in/time_embedder/PDD头/context_embedder）
  DiT 单层 ≈ 1.26GB；resident=2 ⇒ 2.5GB + 槽位 1.26GB + 激活（16MP=16384 token，SDPA）
  编码器：visual+embed+norm 常驻 ≈ 1.5GB，64 层语言层流式（与 DiT 分阶段，峰值不叠加）
  VAE 2.34GB fp32 解码时常驻。默认 resident=2 / enc_resident=0，旗标可调。

用法:
  D:\\ai\\python_h3image\\Scripts\\python h3image_3tier.py --prompt "..." --size 16:9 --seed 42
  ... --ref a.png --ref b.png        # 编辑，顺序 = <Picture 1>...
  ... --dry_run                      # 只打印计划，不加载权重
判活只认 RESULT: 行 + PNG 落盘（长任务铁律）。
"""
import argparse
import json
import os
import queue
import re
import struct
import threading
import time

import torch

DEV = "cuda"
CONVERTED = r"D:\ai\models\MiniMax-H3-Image-diffusers"
ENCODER_SRC = r"D:\ai\models\MiniMax-H3\Ref2VA\text_encoder"
OUT_DIR = r"D:\ai\output\images"
DTYPE = torch.bfloat16

SIZES = {
    "1:1": (4096, 4096), "4:3": (4736, 3552), "3:4": (3552, 4736),
    "3:2": (5024, 3360), "2:3": (3360, 5024), "16:9": (5472, 3072),
    "9:16": (3072, 5472), "21:9": (6272, 2688),
}

# ---------------------------------------------------------------- pread（同 h3_3tier_v2）
_hdr = {}


def _header(sp):
    h = _hdr.get(sp)
    if h is None:
        with open(sp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        _hdr[sp] = h
    return h


def pread(sp, name, dtype=None):
    """dtype=None ⇒ 保留存储精度（fp32 小件不能强转 bf16，官方 _keep_in_fp32_modules 口径）"""
    h = _header(sp)
    m = h[name]
    with open(sp, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + hl + m["data_offsets"][0])
        t = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[m["dtype"]]
        numel = 1
        for s in m["shape"]:
            numel *= s
        buf = f.read(numel * torch.tensor([], dtype=t).element_size())
    out = torch.frombuffer(bytearray(buf), dtype=t).reshape(m["shape"])
    return out if dtype is None else out.to(dtype)


# ---------------------------------------------------------------- 层索引 + LRU（同 h3_3tier_v2 口径）
class LayerIndex:
    """转换后布局：键名=模块路径。按 layer_re 分组，其余归 other。"""

    def __init__(self, tf_dir, layer_re):
        idx = os.path.join(tf_dir, "model.safetensors.index.json")
        wm = json.load(open(idx, encoding="utf-8"))["weight_map"]
        self.layers, self.other = {}, {}
        for name, shard in wm.items():
            sp = os.path.join(tf_dir, shard)
            m = layer_re.match(name)
            if m:
                self.layers.setdefault(int(m.group(1)), {})[name] = sp
            else:
                self.other[name] = sp
        self.n = (max(self.layers) + 1) if self.layers else 0

    def layer_state(self, i):
        return {n: pread(sp, n) for n, sp in self.layers.get(i, {}).items()}


_CACHE = {}
_CACHE_ORDER = []
_LOCK = threading.Lock()


def cache_cap():
    return int(os.environ.get("H3IMG_CACHE_LAYERS", "8"))


def cached_layer_state(index, i):
    with _LOCK:
        hit = _CACHE.get((id(index), i))
    if hit is not None:
        return hit
    try:
        st = index.layer_state(i)
    except MemoryError:
        with _LOCK:
            _CACHE.clear()
            _CACHE_ORDER.clear()
        import gc
        gc.collect()
        print("[h3img_3tier] 内存顶不住 ⇒ 清空层缓存降级现读现用", flush=True)
        st = index.layer_state(i)
    with _LOCK:
        _CACHE[(id(index), i)] = st
        _CACHE_ORDER.append((id(index), i))
        while len(_CACHE_ORDER) > cache_cap():
            old = _CACHE_ORDER.pop(0)
            if old != (id(index), i):
                _CACHE.pop(old, None)
    return st


# ---------------------------------------------------------------- 参数替换（同 h3_3tier_v2）
def _replace_param(module, rel, tensor):
    cur = module
    parts = rel.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    newp = torch.nn.Parameter(tensor, requires_grad=False)
    if last.isdigit():
        cur[int(last)] = newp
    else:
        setattr(cur, last, newp)


def _set_buffer(module, rel, tensor):
    cur = module
    parts = rel.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    if last.isdigit():
        cur._buffers[int(last)] = tensor
    else:
        cur._buffers[last] = tensor


def fill_small(model, other, device=DEV):
    """非层权重：恒等键名直灌（保留存储精度）；buffer 若在 ckpt 里也落位。"""
    state = {n: pread(sp, n) for n, sp in other.items()}
    filled, miss = 0, []
    for name, p in model.named_parameters():
        if name in state:
            _replace_param(model, name, state[name].to(device))
            filled += 1
    for name, buf in list(model.named_buffers()):
        if name in state:
            _set_buffer(model, name, state[name].to(device))
            filled += 1
    return filled, miss


# ---------------------------------------------------------------- 流式层包装
class StreamStack:
    """常驻层真权重上 GPU；流式层参数换 0 字节 CPU 空壳，
    每层 forward 前 pread→槽位→p.data 换入→前向→换回（h3_3tier_v2 ③ 的逐层包装版）。"""

    def __init__(self, blocks, index, resident, tag, pfx, prefetch=True):
        self.blocks, self.index, self.tag, self._pfx = blocks, index, tag, pfx
        self.resident = min(resident, len(blocks))
        n = len(blocks)
        for i in range(self.resident):
            blk = blocks[i]
            blk.to_empty(device=DEV)
            st = cached_layer_state(index, i)
            for name, p in list(blk.named_parameters()):
                full = f"{self._prefix(i)}{name}"
                if full in st:
                    _replace_param(blk, name, st[full].to(device=DEV, dtype=p.dtype))
        for i in range(self.resident, n):
            for name, p in list(blocks[i].named_parameters()):
                _replace_param(blocks[i], name, torch.empty(0, dtype=p.dtype, device="cpu"))
        self.slot = {}
        if self.resident < n:
            real = cached_layer_state(index, self.resident)
            for full, v in real.items():
                rel = full[len(self._prefix(self.resident)):]
                self.slot[rel] = torch.empty(tuple(v.shape), dtype=v.dtype, device=DEV)
        self.q = queue.Queue(maxsize=2)
        if prefetch and self.resident < n:
            threading.Thread(target=self._pref, daemon=True).start()
        torch.cuda.empty_cache()
        print("[h3img_3tier] %s 常驻 %d 层 / 流式 %d 层 / 槽位 %d 个 / 显存 %.2fGB" % (
            tag, self.resident, n - self.resident, len(self.slot),
            torch.cuda.memory_allocated() / 1024 ** 3), flush=True)
        for i in range(self.resident, n):
            self._wrap(i)

    def _prefix(self, i):
        return self._pfx.format(i=i)

    def _pref(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            i, box = it
            box["s"] = cached_layer_state(self.index, i)

    def _wrap(self, i):
        blk = self.blocks[i]
        orig = blk.forward
        stack = self

        def fwd(*a, **kw):
            if stack.prefetch_on:
                box = {}
                stack.q.put((i, box))
                while "s" not in box:
                    threading.Event().wait(0.005)
                st = box["s"]
            else:
                st = cached_layer_state(stack.index, i)
            pfx = stack._prefix(i)
            done = 0
            for full, v in st.items():
                rel = full[len(pfx):]
                if rel in stack.slot:
                    stack.slot[rel].copy_(v.to(stack.slot[rel].dtype))
                    done += 1
            saved = {}
            for name, p in blk.named_parameters():
                saved[name] = p.data
                if name in stack.slot:
                    p.data = stack.slot[name]
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            for name, p in blk.named_parameters():
                p.data = saved[name]
            print("[h3img_3tier] %s layer %d/%d  %.2fs  显存 %.2fGB" % (
                stack.tag, i, len(stack.blocks), time.perf_counter() - t0,
                torch.cuda.memory_allocated() / 1024 ** 3), flush=True)
            return out

        blk.forward = fwd

    prefetch_on = True


# ---------------------------------------------------------------- rope 显式重算
# 非持久 buffer 在 meta 初始化下是 meta 张量，ckpt 里也没有（转换脚本明确 drop rope.inv_freq）
# ⇒ 必须显式重算，否则就是 V4Flash 那次 NaN 的同款根因（脏内存 rope）。
def _fix_dit_rope(model, cfg):
    rf = model.rope.rope_freq_dim
    theta = cfg.get("rope_theta", 10000.0)
    inv = 1.0 / (theta ** (torch.arange(0, 2 * rf, 2, dtype=torch.float32) / (2 * rf)))
    model.rope.inv_freq = inv.to(DEV)


def _fix_encoder_rope(model, cfg):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding)
    lm = model.model.language_model
    fresh = Qwen3VLTextRotaryEmbedding(cfg.text_config)
    for nm, buf in fresh.named_buffers():
        setattr(lm.rotary_emb, nm, buf.to(DEV))
    vc = cfg.vision_config
    head_dim = vc.hidden_size // vc.num_heads
    vfresh = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
    for nm, buf in vfresh.named_buffers():
        setattr(model.model.visual.rotary_pos_emb, nm, buf.to(DEV))


def _assert_no_meta(model, tag):
    left = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    left += [n for n, p in model.named_parameters()
             if p.device.type == "meta" and p.numel() > 0]
    if left:
        raise SystemExit(f"RESULT: FAIL {tag} 仍有 meta 张量 {len(left)} 个: {left[:8]}")


# ---------------------------------------------------------------- 两个大模块的组装
def build_streaming_transformer(resident, cache_layers):
    from diffusers import MiniMaxH3ImageTransformer2DModel
    tf_dir = os.path.join(CONVERTED, "transformer_diffusers")
    cfg = json.load(open(os.path.join(tf_dir, "config.json"), encoding="utf-8"))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with torch.device("meta"):
        model = MiniMaxH3ImageTransformer2DModel(**cfg)
    _fix_dit_rope(model, cfg)
    index = LayerIndex(tf_dir, re.compile(r"^transformer_blocks\.(\d+)\."))
    filled, _ = fill_small(model, index.other)
    print("[h3img_3tier] DiT 小件 %d 个上 GPU（fp32 件保留 fp32）" % filled, flush=True)
    StreamStack(model.transformer_blocks, index, resident, "DiT",
                "transformer_blocks.{i}.")
    _assert_no_meta(model, "DiT")
    return model


def build_streaming_encoder(resident):
    from transformers import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    cfg = Qwen3VLConfig.from_pretrained(ENCODER_SRC)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(cfg)
    _fix_encoder_rope(model, cfg)
    index = LayerIndex(ENCODER_SRC, re.compile(r"^model\.language_model\.layers\.(\d+)\."))
    # visual / embed_tokens / norm / lm_head 归 other：visual+embed+norm 上 GPU，lm_head 留 CPU
    other_gpu = {n: sp for n, sp in index.other.items() if not n.startswith("lm_head.")}
    filled, _ = fill_small(model, other_gpu)
    print("[h3img_3tier] 编码器小件 %d 个上 GPU（visual/embed/norm；lm_head 留 CPU，管线只要 hidden_states[50]）"
          % filled, flush=True)
    StreamStack(model.model.language_model.layers, index, resident, "ENC",
                "model.language_model.layers.{i}.")
    # lm_head 是有意留 meta 的（管线只取 hidden_states[50]，不调 lm_head）⇒ 闸只查非 lm_head
    left = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    left += [n for n, p in model.named_parameters()
             if p.device.type == "meta" and p.numel() > 0 and not n.startswith("lm_head.")]
    if left:
        raise SystemExit(f"RESULT: FAIL ENC 仍有 meta 张量 {len(left)} 个: {left[:8]}")
    return model


# ---------------------------------------------------------------- 主流程
def parse_size(s):
    if s in SIZES:
        return SIZES[s]
    w, h = s.lower().split("x")
    w, h = int(w), int(h)
    if w % 32 or h % 32:
        raise SystemExit(f"RESULT: FAIL 宽高必须是 32 的倍数: {w}x{h}")
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--prompt_file")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--size", default="1:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resident", type=int, default=2, help="DiT 常驻层数")
    ap.add_argument("--enc_resident", type=int, default=0, help="编码器常驻层数")
    ap.add_argument("--cache_layers", type=int, default=8, help="CPU LRU 层缓存上限")
    ap.add_argument("--no_prefetch", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = open(args.prompt_file, encoding="utf-8").read().strip()
    if not prompt:
        raise SystemExit("RESULT: FAIL 需要 --prompt 或 --prompt_file")
    w, h = parse_size(args.size)
    for r in args.ref:
        if not os.path.isfile(r):
            raise SystemExit(f"RESULT: FAIL 参考图不存在: {r}")
    os.environ["H3IMG_CACHE_LAYERS"] = str(args.cache_layers)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(
        OUT_DIR, f"h3img3tier_{time.strftime('%Y%m%d_%H%M%S')}_{args.seed}.png")

    print(f"[plan] converted={CONVERTED}", flush=True)
    print(f"[plan] encoder={ENCODER_SRC}（原生权重直读，不经过转换）", flush=True)
    print(f"[plan] size={w}x{h} seed={args.seed} refs={len(args.ref)} "
          f"resident={args.resident}/{args.enc_resident} cache={args.cache_layers}", flush=True)
    print(f"[plan] steps=6(固定PDD) guidance=无 out={out}", flush=True)
    if args.dry_run:
        print("RESULT: DRY_RUN 未加载权重", flush=True)
        return
    if not os.path.isdir(os.path.join(CONVERTED, "transformer_diffusers")):
        raise SystemExit("RESULT: FAIL 未转换；先跑 D:\\ai\\h3image_convert.py（需主人下令）")

    t0 = time.time()
    import torch as _t
    from diffusers import ComponentsManager, MiniMaxH3ImageBlocks, ModularPipeline
    from diffusers import AutoencoderKLMiniMaxH3Image
    from diffusers.utils import load_image

    transformer = build_streaming_transformer(args.resident, args.cache_layers)
    text_encoder = build_streaming_encoder(args.enc_resident)
    if args.no_prefetch:
        StreamStack.prefetch_on = False
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        os.path.join(CONVERTED, "vae_diffusers"), torch_dtype=_t.float32).to(DEV)

    blocks = MiniMaxH3ImageBlocks()
    manager = ComponentsManager()
    pipe = blocks.init_pipeline(CONVERTED, components_manager=manager)
    # 注入已流式化的大模块；load_components 只补剩下的 tokenizer/processor/scheduler
    pipe.transformer = transformer
    pipe.text_encoder = text_encoder
    pipe.vae = vae
    manager.add("transformer", transformer)
    manager.add("text_encoder", text_encoder)
    manager.add("vae", vae)
    pipe.load_components(dtype=_t.bfloat16)
    print("[load] %.1fs" % (time.time() - t0), flush=True)

    kw = dict(prompt=prompt, height=h, width=w,
              generator=_t.Generator(device="cpu").manual_seed(args.seed),
              output="images")
    if args.ref:
        kw["image"] = [load_image(r) for r in args.ref]
    t1 = time.time()
    img = pipe(**kw)[0]
    img.save(out)
    print(f"RESULT: OK file={out} size={img.size} seed={args.seed} "
          f"gen={time.time()-t1:.1f}s total={time.time()-t0:.1f}s "
          f"gpu_peak={_t.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)


if __name__ == "__main__":
    main()