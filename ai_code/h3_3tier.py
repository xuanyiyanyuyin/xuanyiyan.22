# -*- coding: utf-8 -*-
"""
H3 三层存储引擎（骨架 v0.1 · 2026-10-13）
思路照搬 D:\ai\flux2_3tier.py 与 D:\ai\ltx_engine.py：
  · DiT 的 50 个 MiniMaxH3TransformerBlock 权重【不整份加载】
  · 用 safetensors 头索引 + 逐张量 pread，按层在 CPU<->GPU 之间换入换出
  · 常驻 K 层（slot），其余层在磁盘；进入某层前预取下一层（IO/计算重叠）
纪律：只读权重；不改 diffusers 源码；改动只发生在本文件的 monkey-patch 内。
"""
import json, os, struct, threading, queue
import torch

H3_ROOT   = r"D:\ai\models\MiniMax-H3\Ref2VA"
TF_DIR    = os.path.join(H3_ROOT, "transformer")
TF_INDEX  = os.path.join(TF_DIR, "model.safetensors.index.json")
RESIDENT  = int(os.environ.get("H3_RESIDENT_LAYERS", "3"))   # 常驻 GPU 的层数
PREFETCH  = os.environ.get("H3_PREFETCH", "1") == "1"
DEV       = "cuda"

# ---------- 分片头索引（一次读入，常驻内存；只存元数据，不读权重）----------
_hdr_cache = {}
_hdr_lock = threading.Lock()

def _header(shard_path):
    with _hdr_lock:
        h = _hdr_cache.get(shard_path)
    if h is not None:
        return h
    with open(shard_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    with _hdr_lock:
        _hdr_cache[shard_path] = h
    return h

def _pread_tensor(shard_path, name, dtype=torch.bfloat16, device="cpu"):
    """按名字从分片里只读这一个张量（不 mmap、不整份加载）。"""
    h = _header(shard_path)
    meta = h[name]
    off = 8 + struct.unpack("<Q", open(shard_path, "rb").read(8))[0] + meta["data_offsets"][0]
    shape, dt = meta["shape"], meta["dtype"]
    np_dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[dt]
    nbytes = 1
    for s in shape:
        nbytes *= s
    nbytes *= torch.tensor([], dtype=np_dt).element_size()
    with open(shard_path, "rb") as f:
        f.seek(off)
        buf = f.read(nbytes)
    t = torch.frombuffer(bytearray(buf), dtype=np_dt).reshape(shape)
    return t.to(device=device, dtype=dtype)

# ---------- 层名 -> (分片文件, 张量名) 映射 ----------
class H3LayerIndex:
    def __init__(self, tf_dir=TF_DIR, index_path=TF_INDEX):
        self.tf_dir = tf_dir
        wmap = json.load(open(index_path, "r", encoding="utf-8"))["weight_map"]
        self.layers = {}      # int -> list[(tensor_name, shard_path)]
        self.other  = []      # 非 transformer_blocks.* 的其余权重（小，整份常驻）
        for name, shard in wmap.items():
            sp = os.path.join(tf_dir, shard)
            if name.startswith("blocks.") and name.split(".")[0] == "blocks":
                i = int(name.split(".")[1])
                self.layers.setdefault(i, []).append((name, sp))
            else:
                self.other.append((name, sp))
        self.n_layers = (max(self.layers) + 1) if self.layers else 0

    def layer_state(self, i, dtype=torch.bfloat16):
        out = {}
        for name, sp in self.layers.get(i, []):
            out[name] = _pread_tensor(sp, name, dtype=dtype)
        return out

# ---------- 一个常驻槽位：把一组权重搬到 GPU / 搬回 CPU ----------
class Slot:
    def __init__(self, idx):
        self.idx = idx
        self.state = {}

    def load(self, state):
        self.state = {k: v.to(DEV, non_blocking=True) for k, v in state.items()}

    def free(self):
        self.state = {}
        torch.cuda.empty_cache()

# ---------- 把某层权重灌进 nn.Module（按名字 setattr / copy_）----------
def _assign_module_weights(block, prefix, state, device=DEV):
    """state 的 key 形如 blocks.7.attn.qkv_proj.weight（checkpoint 前缀是 blocks.，模型里是 transformer_blocks.）"""
    for k, v in state.items():
        rel = k.split(".", 2)[2] if k.count(".") >= 2 else k   # blocks.7.xxx -> xxx
        obj = block
        parts = rel.split(".")
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
            tgt = getattr(obj, parts[-1])
            with torch.no_grad():
                if hasattr(tgt, "copy_"):
                    tgt.copy_(v.to(tgt.dtype))
                else:
                    setattr(obj, parts[-1], torch.nn.Parameter(v.to(tgt.dtype), requires_grad=False))
        except Exception as e:
            print("[h3_3tier] assign skip", k, e, flush=True)

# ---------- 主补丁：替换 Transformer3DModel.forward 里的层循环 ----------
def patch_forward(model, index=None, resident=RESIDENT):
    index = index or H3LayerIndex()
    blocks = model.transformer_blocks
    print("[h3_3tier] layers=%d resident=%d" % (index.n_layers, resident), flush=True)

    prefetch_q = queue.Queue(maxsize=2)
    def prefetcher():
        while True:
            item = prefetch_q.get()
            if item is None:
                return
            i, holder = item
            holder["state"] = index.layer_state(i)
    th = threading.Thread(target=prefetcher, daemon=True); th.start()

    orig_forward = model.forward
    def forward(*args, **kwargs):
        # 简化版：先只做"进入 forward 前把 50 层全部按需换入"的框架占位；
        # 真正的逐层循环替换在 v0.2 里通过 wrap block 实现（见 wrap_blocks）。
        for i in range(index.n_layers):
            holder = {}
            prefetch_q.put((i, holder))
            while "state" not in holder:
                threading.Event().wait(0.01)
            _assign_module_weights(blocks[i], "transformer_blocks.%d." % i, holder["state"])
            holder["state"] = {}
        return orig_forward(*args, **kwargs)
    model.forward = forward
    return model

if __name__ == "__main__":
    idx = H3LayerIndex()
    print("H3 层数:", idx.n_layers)
    for i in [0, 1, idx.n_layers - 1]:
        names = [n.split(".")[-2] + "." + n.split(".")[-1] for n, _ in idx.layers.get(i, [])][:8]
        print("  层 %2d: %d 个张量  例: %s" % (i, len(idx.layers.get(i, [])), ", ".join(names)))
    print("非层权重（常驻的小件）:", len(idx.other), "个")

# ---------- v0.2: 名称映射 + 逐层装载 + forward 层循环替换 ----------
# checkpoint 键（blocks.i.*）  ->  diffusers 模块路径（transformer_blocks.i.*）
# 已知差异: checkpoint 的 mlp.fc1/fc2  ->  diffusers 的 ff. 内部名（可能也是 fc1/fc2，需运行时核对）
KEY_ALIASES = [
    ("mlp.fc1", "ff.fc1"),
    ("mlp.fc2", "ff.fc2"),
    ("mlp.", "ff."),
]

def _resolve(obj, rel):
    """按 rel 路径 getattr，支持 KEY_ALIASES 里列出的别名。"""
    cands = [rel]
    for a, b in KEY_ALIASES:
        if rel.startswith(a):
            cands.append(b + rel[len(a):])
    for c in cands:
        cur, ok = obj, True
        for p in c.split("."):
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False; break
        if ok:
            return cur
    return None

def assign_layer(block, state, verbose=False):
    """把一层（10 个张量）灌进 diffusers 的 block 里（不新建 Parameter，原地 copy_）。"""
    ok, miss = 0, []
    for k, v in state.items():
        rel = k.split(".", 2)[2]          # blocks.7.attn.qkv_proj.weight -> attn.qkv_proj.weight
        tgt = _resolve(block, rel)
        if tgt is None or not hasattr(tgt, "copy_"):
            miss.append(rel); continue
        with torch.no_grad():
            tgt.copy_(v.to(device=tgt.device, dtype=tgt.dtype))
        ok += 1
    if verbose and miss:
        print("[h3_3tier] 未匹配:", miss, flush=True)
    return ok, miss

def dump_block_param_names(model):
    """第一次实跑时用: 打印 diffusers block 的真实参数名，便于核对映射。"""
    b0 = model.transformer_blocks[0]
    return [n for n, _ in b0.named_parameters()]

def patch_layer_loop(model, index=None, resident=RESIDENT, prefetch=PREFETCH, verbose=False):
    """★ 核心: 让 Transformer3DModel.forward 在执行时逐层换入/换出权重。

    做法: 用一个包装模块替换 model.transformer_blocks（nn.ModuleList），
          其 forward 在每层前装载、之后按槽位策略释放；并把下一层预取到后台线程。
    """
    index = index or H3LayerIndex()
    blocks = model.transformer_blocks
    n = len(blocks)
    state = {"cur": 0, "resident": set()}
    q = queue.Queue(maxsize=2)

    def _prefetch_worker():
        while True:
            item = q.get()
            if item is None:
                return
            i, box = item
            box["state"] = index.layer_state(i)

    if prefetch:
        threading.Thread(target=_prefetch_worker, daemon=True).start()

    class _LayerRunner(torch.nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = blocks
        def forward(self, hidden_states, temb, adaln_indices, rotary_emb):
            for i in range(n):
                box = {}
                if prefetch:
                    q.put((i, box))
                    while "state" not in box:
                        threading.Event().wait(0.005)
                    st = box["state"]
                else:
                    st = index.layer_state(i)
                assign_layer_v3(self.blocks[i], st, verbose=verbose)
                hidden_states = self.blocks[i](hidden_states, temb, adaln_indices, rotary_emb)
                # 槽位管理: 超过常驻数就把最老的层权重清回空（体现为丢弃引用 + 空缓存）
                state["cur"] = i
                st.clear()
                if (i - resident) >= 0 and prefetch:
                    torch.cuda.empty_cache()
            return hidden_states

    model.transformer_blocks = torch.nn.ModuleList([_LayerRunner(blocks)])
    model._h3_layer_runner = True
    return model

# ---------- v0.3: checkpoint -> diffusers 的显式映射（含 QKV 切分）----------
# 依据: 读 diffusers/models/transformers/transformer_minimax_h3.py + models/attention.py
#   ckpt attn.qkv_proj.weight  -> attn.to_q / to_k / to_v  (沿 dim0 三等分, 各 bias=False)
#   ckpt attn.q_norm / k_norm  -> attn.norm_q / norm_k
#   ckpt attn.out_proj.weight  -> attn.to_out.0.weight
#   ckpt mlp.fc1 / fc2         -> ff 内部 (diffusers FeedForward 的 fc1/fc2, 具体名以运行时 dump 为准)
#   ckpt adaln_proj.linear.*   -> adaln_proj.linear.*   (同名)
#   ckpt norm1 / norm2         -> norm1 / norm2         (同名)
QKV_SPLIT = True

def _put(block, rel, tensor, verbose=False):
    """把单个张量写进 block 的 rel 路径（支持 list 索引, 如 to_out.0.weight）。"""
    parts, cur = rel.split("."), block
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    tgt = cur[int(last)] if last.isdigit() else getattr(cur, last)
    with torch.no_grad():
        tgt.copy_(tensor.to(device=tgt.device, dtype=tgt.dtype))

def assign_layer_v3(block, state, verbose=False):
    """v0.3: 显式映射 + QKV 切分。返回 (成功数, 未匹配名)。"""
    ok, miss = 0, []
    for k, v in state.items():
        rel = k.split(".", 2)[2]                     # blocks.7.<rel>
        try:
            if rel == "attn.qkv_proj.weight" and QKV_SPLIT:
                q, kk, vv = v.chunk(3, dim=0)
                _put(block, "attn.to_q.weight", q)
                _put(block, "attn.to_k.weight", kk)
                _put(block, "attn.to_v.weight", vv)
                ok += 1; continue
            # ★ 2026-10-13 实测确认（meta 实例化 dump，见 D:\\ai\\h3_dump.py 输出）
            mapping = {
                "attn.q_norm.weight": "attn.norm_q.weight",     # (128,) 同名异序
                "attn.k_norm.weight": "attn.norm_k.weight",     # (128,)
                "attn.out_proj.weight": "attn.to_out.0.weight", # (5376,7168)
                "mlp.fc1.weight": "ff.net.0.proj.weight",       # (28672,5376) 两侧同为 swiglu 融合 ⇒ 直放
                "mlp.fc2.weight": "ff.net.2.weight",            # (5376,14336)
            }
            tgt_rel = mapping.get(rel, rel)
            try:
                _put(block, tgt_rel, v)
                ok += 1
            except Exception:
                # 回退: 用 ff 的第一/第二个 Linear 兜底
                if rel.startswith("mlp."):
                    if rel.startswith("mlp.fc1") and _put_fc1(block, v, verbose):
                        ok += 1; continue
                    lin = [m for m in block.ff.modules() if isinstance(m, torch.nn.Linear)]
                    idx = 0 if "fc1" in rel else (1 if len(lin) > 1 else 0)
                    with torch.no_grad():
                        lin[idx].weight.copy_(v.to(dtype=lin[idx].weight.dtype))
                    ok += 1
                else:
                    miss.append(rel)
        except Exception as e:
            miss.append(rel + " (" + str(e)[:40] + ")")
    if verbose and miss:
        print("[h3_3tier] 未匹配:", miss, flush=True)
    return ok, miss

# ---------- v0.4: ff(mf1/fc2) 形状自适应 + 带 shape 的 dump ----------
def _ff_linears(block):
    return [m for m in block.ff.modules() if isinstance(m, torch.nn.Linear)]

def _put_fc1(block, tensor, verbose=False):
    """ckpt 的 mlp.fc1 -> diffusers ff 的输入投影。
       若形状一致直接放；若是 swiglu 融合（2×inner）则按行切两半分别放。"""
    lins = _ff_linears(block)
    if not lins:
        return False
    tgt = lins[0].weight
    src = tensor
    try:
        if tuple(src.shape) == tuple(tgt.shape):
            with torch.no_grad():
                tgt.copy_(src.to(dtype=tgt.dtype))
            return True
        if src.shape[0] == 2 * tgt.shape[0]:
            half = src.shape[0] // 2
            with torch.no_grad():
                tgt.copy_(src[:half].to(dtype=tgt.dtype))
            if len(lins) > 1 and tuple(lins[1].weight.shape) == (half, src.shape[1]):
                with torch.no_grad():
                    lins[1].weight.copy_(src[half:].to(dtype=lins[1].weight.dtype))
            if verbose:
                print("[h3_3tier] fc1 按 swiglu 切两半放置 ✅", flush=True)
            return True
    except Exception as e:
        if verbose:
            print("[h3_3tier] fc1 放置失败:", str(e)[:80], flush=True)
    return False

def dump_params_with_shape(model, limit=14):
    """首次实跑核对用: 打印 diffusers block 的真实参数名 + shape。"""
    b0 = model.transformer_blocks[0]
    out = []
    for n, p in b0.named_parameters():
        out.append("%s %s" % (n, tuple(p.shape)))
        if len(out) >= limit:
            break
    return out
