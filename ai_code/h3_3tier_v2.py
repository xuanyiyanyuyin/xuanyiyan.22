# -*- coding: utf-8 -*-
"""H3 DiT 流式引擎 v1.1（真实可用版）
  · 小件 + 常驻层: 从磁盘 pread 真权重 ⇒ 直接放 GPU
  · 其余 48 层: 参数换成 CPU 空壳（meta 不能 .to() ⇒ 必须换掉）
  · 每层用前: pread → 灌 GPU 单槽位 → 换 p.data 指针 → 前向 → 换回
  · 重写 MiniMaxH3Transformer3DModel.forward 以插入搬运（同 flux2_3tier）
"""
import os, json, struct, threading, queue, time
import torch

DEV = "cuda"
H3_ROOT = r"D:\ai\models\MiniMax-H3\Ref2VA"
TF_DIR = os.path.join(H3_ROOT, "transformer")
TF_INDEX = os.path.join(TF_DIR, "model.safetensors.index.json")
DTYPE = torch.bfloat16

_hdr = {}
def _header(sp):
    h = _hdr.get(sp)
    if h is None:
        with open(sp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        _hdr[sp] = h
    return h

def pread(sp, name, dtype=DTYPE):
    h = _header(sp)
    m = h[name]
    hl = struct.unpack("<Q", open(sp, "rb").read(8))[0]
    off = 8 + hl + m["data_offsets"][0]
    shape = m["shape"]
    t = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[m["dtype"]]
    numel = 1
    for s in shape:
        numel *= s
    nb = numel * torch.tensor([], dtype=t).element_size()
    with open(sp, "rb") as f:
        f.seek(off)
        buf = f.read(nb)
    return torch.frombuffer(bytearray(buf), dtype=t).reshape(shape).to(dtype)

# ★ 小件（非层）的命名对照（ckpt → diffusers）
# 方向: (diffusers 前缀, ckpt 前缀)  ★ 2026-10-14 修正方向 ✅
SMALL_ALIAS = [
    # ★ 2026-10-14 全名单对照（ckpt 35 个非层权重）后补全
    ("proj_in.", "video_patch_proj."),
    ("audio_proj_in.", "audio_patch_proj."),
    ("context_embedder.", "condition_proj."),
    ("time_embedder.linear_1.", "time_embedder.proj_in."),
    ("time_embedder.linear_2.", "time_embedder.proj_out."),
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
    ("token_refiner.norm_out.", "token_refiner.final_norm."),
    ("token_refiner.final_norm.", "token_refiner.final_norm."),
    ("norm_out.adaln_proj.linear.", "final_layer.adaln_proj.linear."),
    ("norm_out.linear.", "final_layer.adaln_proj.linear."),   # ★ 最后 2 个 ✅
    ("norm_out.norm.", "final_layer.norm."),
    ("norm_out.", "final_layer.norm."),
    ("proj_out.", "final_layer.video_out."),
    ("audio_proj_out.", "final_layer.audio_out."),
]
# 层内别名（diffusers 名 → ckpt 名），层和小件共用
INNER_ALIAS = [
    ("attn.to_out.0.", "attn.out_proj."),
    ("attn.norm_q.", "attn.q_norm."),
    ("attn.norm_k.", "attn.k_norm."),
    ("ff.net.0.proj.", "mlp.fc1."),
    ("ff.net.2.", "mlp.fc2."),
]

def ckpt_name_for(diffusers_name):
    """把 diffusers 的参数名翻译成 checkpoint 里的名字（多层回退）。"""
    cands = []
    def inner_variants(n):
        out = [n]
        for a, b in INNER_ALIAS:
            if a in n:
                out.append(n.replace(a, b))
        return out
    # 1) 直接把层内别名换掉
    cands += inner_variants(diffusers_name)
    # 2) 再把小件前缀换掉
    for dv, ck in SMALL_ALIAS:
        for base in list(cands):
            if base.startswith(dv):
                cands.append(ck + base[len(dv):])
                cands += inner_variants(ck + base[len(dv):])
    # 3) qkv 特例: 对【每个已生成的候选】都追加 qkv 变体（含前缀翻译后的 ✅）
    for base in list(cands):
        for q in ("attn.to_q.", "attn.to_k.", "attn.to_v."):
            if q in base:
                cands.append(base.split(q)[0] + "attn.qkv_proj.weight")   # ★ 必须带 weight ✅
    # 4) qkv 需要三等分 ⇒ 在填充处特殊处理（见 _fill_module 的 qkv 分支）
    return cands

RENAME = {
    "attn.q_norm.weight": "attn.norm_q.weight",
    "attn.k_norm.weight": "attn.norm_k.weight",
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "mlp.fc1.weight": "ff.net.0.proj.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
}

_CACHE = {}
_CACHE_ORDER = []
CACHE_MAX = int(os.environ.get("H3_CACHE_LAYERS", "40"))  # ★ 尽量全放（靠页文件 ✅）· 40 层 ≈ 49 GB
_LOCK = threading.Lock()

class H3LayerIndex:
    def __init__(self):
        wm = json.load(open(TF_INDEX, encoding="utf-8"))["weight_map"]
        self.layers, self.other = {}, []
        for name, shard in wm.items():
            rec = (name, os.path.join(TF_DIR, shard))
            if name.startswith("blocks."):
                self.layers.setdefault(int(name.split(".")[1]), []).append(rec)
            else:
                self.other.append(rec)
        self.n = (max(self.layers) + 1) if self.layers else 0

    def layer_state(self, i, dtype=DTYPE):
        with _LOCK:
            hit = _CACHE.get(i)
        if hit is not None:
            return hit
        try:
            st = {n: pread(sp, n, dtype) for n, sp in self.layers.get(i, [])}
        except MemoryError:
            # ★ 优雅降级: 清空缓存重试（内存/页文件顶不住时退化成"现读现用" ✅）
            with _LOCK:
                _CACHE.clear(); _CACHE_ORDER.clear()
            import gc as _gc; _gc.collect()
            print("[h3_3tier] ⚠️ 内存顶不住 ⇒ 清空层缓存后重试（降级为现读现用）", flush=True)
            st = {n: pread(sp, n, dtype) for n, sp in self.layers.get(i, [])}
        with _LOCK:
            _CACHE[i] = st
            _CACHE_ORDER.append(i)
            while len(_CACHE_ORDER) > CACHE_MAX:          # ★ LRU 淘汰，防止内存爆 ✅
                old = _CACHE_ORDER.pop(0)
                if old != i:
                    _CACHE.pop(old, None)
        return st

def _set_buffer(module, rel, tensor):
    """按点分路径设置 buffer（meta → 真数据）✅"""
    cur = module
    parts = rel.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    if last.isdigit():
        cur._buffers[int(last)] = tensor
    else:
        cur._buffers[last] = tensor

def _replace_param(block, rel, tensor):
    """★ 直接把参数对象换掉（meta 张量不能用 .data 赋值 ✅）"""
    cur = block
    parts = rel.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    newp = torch.nn.Parameter(tensor, requires_grad=False)
    if last.isdigit():
        cur[int(last)] = newp
    else:
        setattr(cur, last, newp)

def _set_param(block, rel, tensor):
    cur = block
    parts = rel.split(".")
    for p in parts[:-1]:
        cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
    last = parts[-1]
    if last.isdigit():
        cur[int(last)].data = tensor
    else:
        getattr(cur, last).data = tensor

def _fill_module(module, prefix, state, device=DEV, verbose=False, miss=None):
    """按 diffusers 名 → ckpt 名 的模糊匹配填补权重（meta 也能填 ✅）。"""
    cnt = 0
    for name, p in module.named_parameters():
        full = prefix + name
        t = None
        for cand in ckpt_name_for(full):
            if cand in state:
                t = state[cand]; break
        if t is None:
            if miss is not None:
                miss.append(full)
            continue
        # ★ qkv 三等分（to_q/to_k/to_v 共享一个 qkv_proj 张量）
        if ("attn.to_q." in full or "attn.to_k." in full or "attn.to_v." in full) and t.shape[0] == 3 * p.shape[0]:
            idx = 0 if "to_q" in full else (1 if "to_k" in full else 2)
            t = t.chunk(3, dim=0)[idx]
        if tuple(t.shape) != tuple(p.shape):
            if miss is not None:
                miss.append(full + " shape" + str(tuple(t.shape)) + "!=" + str(tuple(p.shape)))
            continue
        _replace_param(module, name, t.to(device=device, dtype=p.dtype))
        cnt += 1
    return cnt

def _fill_layer(block, state, device=DEV):
    """把一层的 ckpt 权重（blocks.i.*）灌进 diffusers block（含 qkv 切分与别名）。"""
    st = {k.split(".", 2)[2]: v for k, v in state.items()}
    qkv = st.pop("attn.qkv_proj.weight", None)
    if qkv is not None:
        q, kk, vv = qkv.chunk(3, dim=0)
        _replace_param(block, "attn.to_q.weight", q.to(device))
        _replace_param(block, "attn.to_k.weight", kk.to(device))
        _replace_param(block, "attn.to_v.weight", vv.to(device))
    for rel, v in st.items():
        tgt = RENAME.get(rel, rel)
        try:
            _replace_param(block, tgt, v.to(device))
        except Exception:
            pass

def assign_into_slot(block, state, slot):
    done = 0
    for k, v in state.items():
        rel = k.split(".", 2)[2]
        if rel == "attn.qkv_proj.weight":
            q, kk, vv = v.chunk(3, dim=0)
            for nm, part in (("attn.to_q.weight", q), ("attn.to_k.weight", kk), ("attn.to_v.weight", vv)):
                slot[nm].copy_(part); done += 1
            continue
        rel2 = RENAME.get(rel, rel)
        if rel2 in slot:
            slot[rel2].copy_(v); done += 1
    return done

def patch_h3_forward(model, index=None, resident=2, prefetch=True, verbose=True):
    index = index or H3LayerIndex()
    blocks = model.transformer_blocks
    n = len(blocks)
    # ① 小件: 真权重 → GPU
    others = {nm: pread(sp, nm, DTYPE) for nm, sp in index.other}
    filled, miss = 0, []
    for nm, mod in (("proj_in", model.proj_in), ("audio_proj_in", model.audio_proj_in),
                    ("context_embedder", model.context_embedder), ("time_embedder", model.time_embedder),
                    ("token_refiner", model.token_refiner), ("norm_out", model.norm_out),
                    ("proj_out", model.proj_out), ("audio_proj_out", model.audio_proj_out)):
        try:
            mod.to_empty(device=DEV)
        except Exception:
            try:
                mod.to(DEV)
            except Exception:
                pass
        filled += _fill_module(mod, nm + ".", others, verbose=verbose, miss=miss)
    # ①b ★ buffers（不是 parameters！rope.inv_freq 等）也要落位，否则 rope 计算会撞 meta ✅
    fixed_buf = 0
    for nm, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        t = others.get(nm)
        if t is None:
            for cand in ckpt_name_for(nm):
                if cand in others:
                    t = others[cand]; break
        if t is not None:
            try:
                _set_buffer(model, nm, t.to(device=DEV))   # ★ buffer 也要在 GPU ✅
                fixed_buf += 1
            except Exception:
                pass
    # ② 常驻层: 真权重 → GPU
    for i in range(min(resident, n)):
        blk = blocks[i]
        try:
            blk.to_empty(device=DEV)
        except Exception:
            blk.to(DEV)
        st = index.layer_state(i)
        _fill_layer(blk, st, DEV)
    # ③ 流式层: 参数换成 CPU 空壳
    for i in range(min(resident, n), n):
        for name, p in list(blocks[i].named_parameters()):
            _replace_param(blocks[i], name, torch.empty(0, dtype=p.dtype, device="cpu"))   # ★ 0 字节占位（用前必被槽位换掉 ✅）
    slot = {}
    if min(resident, n) < n:
        # ★ 用【真实 shape】建槽位（流式层参数已换成 0 字节空壳，不能拿它的 shape ✅）
        _real = index.layer_state(min(resident, n))
        for _k, _v in _real.items():
            _rel = _k.split(".", 2)[2]
            if _rel == "attn.qkv_proj.weight":
                _q, _kk, _vv = _v.chunk(3, dim=0)
                slot["attn.to_q.weight"] = torch.empty(tuple(_q.shape), dtype=DTYPE, device=DEV)
                slot["attn.to_k.weight"] = torch.empty(tuple(_kk.shape), dtype=DTYPE, device=DEV)
                slot["attn.to_v.weight"] = torch.empty(tuple(_vv.shape), dtype=DTYPE, device=DEV)
            else:
                slot[RENAME.get(_rel, _rel)] = torch.empty(tuple(_v.shape), dtype=DTYPE, device=DEV)
        print("[h3_3tier] 槽位张量 %d 个（真实 shape ✅）" % len(slot), flush=True)
    torch.cuda.empty_cache()
    if verbose:
        print("[h3_3tier] 小件 %d 个 ✅ · buffer 补 %d 个 ✅ · 常驻 %d 层 ✅ · 流式 %d 层 · 槽位 %d · 显存 %.2fGB" % (
            filled, fixed_buf, resident, n - resident, len(slot), torch.cuda.memory_allocated() / 1024 ** 3), flush=True)
    if miss:
        print("[h3_3tier] ⚠️ 未匹配 %d 个（会留 meta）: %s" % (len(miss), miss[:8]), flush=True)
    q = queue.Queue(maxsize=2)
    def _pref():
        while True:
            it = q.get()
            if it is None:
                return
            i, box = it
            box["s"] = index.layer_state(i)
    if prefetch:
        threading.Thread(target=_pref, daemon=True).start()
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
    def forward(hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                token_tags, position_ids, video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True, **kw):
        # ★ 输入统一搬到 GPU（条件来自 CPU 上的 VAE/编码器 ✅）
        hidden_states = hidden_states.to(DEV, DTYPE)          # ★ 同时统一 dtype ✅
        audio_hidden_states = audio_hidden_states.to(DEV, DTYPE)
        encoder_hidden_states = encoder_hidden_states.to(DEV, DTYPE)
        timestep = timestep.to(DEV, torch.float32)
        timestep_indices = timestep_indices.to(DEV)
        token_tags = token_tags.to(DEV)
        position_ids = position_ids.to(DEV)
        video_indices = video_indices.to(DEV)
        audio_indices = audio_indices.to(DEV)
        text_indices = text_indices.to(DEV)
        r = model.rope(position_ids)
        ve = model.proj_in(hidden_states.to(model.proj_in.weight.dtype))
        ae = model.audio_proj_in(audio_hidden_states.to(model.audio_proj_in.weight.dtype))
        te = model.context_embedder(encoder_hidden_states.to(model.context_embedder.weight.dtype))
        te = model.token_refiner(te)
        h = te.new_zeros((te.shape[0], position_ids.shape[0], te.shape[-1]))
        h = h.index_copy(1, text_indices, te)
        h = h.index_copy(1, video_indices, ve.to(te.dtype))
        h = h.index_copy(1, audio_indices, ae.to(te.dtype))
        temb = model.time_embedder(model.time_proj(timestep).to(model.time_embedder.linear_1.weight.dtype))
        adaln_idx = timestep_indices * 3 + token_tags
        t0 = time.perf_counter()
        for i in range(n):
            blk = blocks[i]
            if i < resident:
                h = blk(h, temb, adaln_idx, r)
                continue
            if prefetch:
                box = {}
                q.put((i, box))
                while "s" not in box:
                    threading.Event().wait(0.005)
                st = box["s"]
            else:
                st = index.layer_state(i)
            assign_into_slot(blk, st, slot)
            saved = {}
            for name, p in blk.named_parameters():
                saved[name] = p.data
                p.data = slot[name]
            h = blk(h, temb, adaln_idx, r)
            for name, p in blk.named_parameters():
                p.data = saved[name]
            if verbose:                      # ★ 每层都打印（主人 2026-10-14 令: 要能一眼看出卡没卡 ✅）
                print("[h3_3tier] layer %d/%d  %.1fs  显存 %.2fGB" % (
                    i, n, time.perf_counter() - t0, torch.cuda.memory_allocated() / 1024 ** 3), flush=True)
        h = model.norm_out(h, temb, timestep_indices).to(model.proj_out.weight.dtype)
        vo = model.proj_out(h).index_select(1, video_indices)
        ao = model.audio_proj_out(h).index_select(1, audio_indices)
        # ★ 按输入设备返回（pipeline 内部在 CPU ⇒ 输出回 CPU；DiT 内部才用 GPU ✅）
        _dev = hidden_states.device if hidden_states.device.type == "cpu" else torch.device("cpu")
        vo = vo.to(_dev); ao = ao.to(_dev)
        return MiniMaxH3TransformerOutput(sample=vo, audio_sample=ao) if return_dict else (vo, ao)
    model.forward = forward
    return model
