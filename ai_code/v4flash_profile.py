# -*- coding: utf-8 -*-
"""V4Flash 精准剖析: 一个 token 从输入到输出每一环节耗时 (抛开冷启动)
monkey-patch 计时: 路由/专家读取等待/GEMM/层加载H2D/注意力/MTP/缓存命中统计
"""
import sys, time, torch, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer

T = collections.defaultdict(float)   # 累积耗时 (秒)
C = collections.defaultdict(int)     # 次数
HIT = collections.defaultdict(int)   # 专家缓存命中统计

# ─── 1. V4LazyMoE.forward 计时副本 (router / read_wait / gemm) ───
_orig_lazy = vs.V4LazyMoE.forward
def timed_lazy(self, hidden_states, input_ids=None):
    B, S, H = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.reshape(-1, H)
    t0 = time.time()
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    T['moe_router'] += time.time() - t0
    C['moe_calls'] += 1

    uids = indices.unique().tolist()
    if self.mtp:
        futs = {eid: self.engine.expert_pool.submit(self.engine.get_mtp_expert, eid) for eid in uids}
    else:
        pool = self.engine.pool_for_lid(self.lid)
        futs = {eid: pool.submit(self.engine.get_expert, self.lid, eid) for eid in uids}
    out = torch.zeros(flat.shape[0], H, device=flat.device, dtype=torch.float32)
    t_r0 = time.time()
    results = {eid: futs[eid].result() for eid in uids}   # 主线程等待专家读取
    T['moe_read_wait'] += time.time() - t_r0
    T['moe_experts'] += len(uids)
    C['moe_experts'] += len(uids)

    t_g0 = time.time()
    for eid in uids:
        w1, s1, w2, s2, w3, s3 = results[eid]
        hit_2d = (indices == eid)
        tidx = hit_2d.any(-1).nonzero().squeeze(-1)
        if tidx.numel() == 0:
            continue
        x = flat[tidx]
        dev = x.device
        g = vs.FP8_KERNEL.matmul_2d(x, w1.to(dev), s1.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        u = vs.FP8_KERNEL.matmul_2d(x, w3.to(dev), s3.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        gate = g.clamp(max=self.limit)
        up = u.clamp(min=-self.limit, max=self.limit)
        inter = torch.nn.functional.silu(gate) * up
        d = vs.FP8_KERNEL.matmul_2d(inter, w2.to(dev), s2.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        w = (weights * hit_2d).sum(-1)[tidx]
        out[tidx] += d * w.unsqueeze(-1)
    T['moe_gemm'] += time.time() - t_g0

    t_s0 = time.time()
    shared = self.shared_experts(residual).to(torch.float32)
    ret = (out.view(B, S, H) + shared).to(hidden_states.dtype)
    T['moe_shared'] += time.time() - t_s0
    return ret
vs.V4LazyMoE.forward = timed_lazy

# ─── 2. ExpertLRU.get: RAM命中 vs 磁盘miss 计时 ───
_orig_lru = vs.ExpertLRU.get
def timed_lru(self, lid, eid):
    key = (lid, eid)
    with self.lock:
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
    if hit is not None:
        HIT['ram_hit'] += 1
        return hit
    t0 = time.time()
    r = _orig_lru(self, lid, eid)
    T['lru_disk'] += time.time() - t0
    HIT['disk_miss'] += 1
    return r
vs.ExpertLRU.get = timed_lru

# ─── 3. get_expert: GPU命中计数 + 总耗时 ───
_orig_get = vs.V4FlashEngine.get_expert
def timed_get(self, lid, eid):
    key = (lid, eid)
    with self.exp_lock:
        gh = self.gpu_experts.get(key)
    if gh is not None:
        HIT['gpu_hit'] += 1
    t0 = time.time()
    r = _orig_get(self, lid, eid)
    T['get_expert'] += time.time() - t0
    return r
vs.V4FlashEngine.get_expert = timed_get

# ─── 4. 层加载 H2D (build_pool) ───
_orig_build = vs.V4FlashEngine._build_and_upload
def timed_build(self, lid):
    t0 = time.time()
    r = _orig_build(self, lid)
    T['build_h2d'] += time.time() - t0
    C['builds'] += 1
    return r
vs.V4FlashEngine._build_and_upload = timed_build

# ─── 5. 层 forward 总耗时 (attention + MoE) ───
_orig_layer = DeepseekV4DecoderLayer.forward
def timed_layer(self, hidden_states, **kw):
    t0 = time.time()
    r = _orig_layer(self, hidden_states, **kw)
    T['layer_total'] += time.time() - t0
    C['layers'] += 1
    return r
DeepseekV4DecoderLayer.forward = timed_layer

# ─── 6. MTP 模块 ───
_orig_mtp = vs.V4MTPModule.forward
def timed_mtp(self, token_ids, hidden_states, position_ids, cache):
    t0 = time.time()
    r = _orig_mtp(self, token_ids, hidden_states, position_ids, cache)
    T['mtp_fwd'] += time.time() - t0
    C['mtp_calls'] += 1
    return r
vs.V4MTPModule.forward = timed_mtp

# ─── 7. 主 forward 总耗时 (记录每次的 token 数) ───
FWD = []   # (token数, 耗时)
_orig_fwd = vs.V4FlashEngine.forward
def timed_fwd(self, input_ids, past_key_values=None, return_hidden=False):
    t0 = time.time()
    r = _orig_fwd(self, input_ids, past_key_values, return_hidden)
    dt = time.time() - t0
    T['main_fwd'] += dt
    C['fwds'] += 1
    FWD.append((input_ids.shape[1], dt))
    return r
vs.V4FlashEngine.forward = timed_fwd

# ═══ 运行 ═══
print('[prof] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok_ = __import__('transformers').AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
text = ("DeepSeek-V4-Flash 是一款高效的多专家混合架构大语言模型，"
        "采用了滑动窗口注意力与压缩稀疏注意力的混合设计，"
        "在保持高性能的同时大幅降低计算成本。"
        "该模型通过 MQA 共享键值机制减少内存占用，"
        "并使用分层缓存将权重分布在 GPU、内存与磁盘之间。"
        "推理引擎采用逐层预读取与专家按需加载策略。") * 3
ids = tok_(text, return_tensors='pt').input_ids[:, :64].to(vs.DEVICE)
N = ids.shape[1]

# 预热: prefill + 4 轮 MTP (排掉冷启动/页缓存冷)
print('[prof] 预热 (prefill + 4轮)...', flush=True)
eng.forward(ids)                      # 冷 prefill
eng.generate(ids[:, :1], max_new_tokens=4, do_sample=False)  # 4 轮热手

# 正式: 先 prefill (不计时), 再手动 MTP decode 轮 (纯计时)
print('[prof] 正式: prefill(不计) + 6 轮 MTP decode...', flush=True)
logits, cache, h = eng.forward(ids, return_hidden=True)   # prefill, 不计入计时
# 清空所有累加器 (含 FWD)
for k in T: T[k] = 0
for k in C: C[k] = 0
for k in HIT: HIT[k] = 0
FWD.clear()
t0 = time.time()
n_gen = 0
for _ in range(6):
    # 单池投机预取 (照 generate 路径)
    if eng._last_layer_experts:
        eng.prefetch_prev_experts()
    last = logits[:, -1, :].clone().float()
    nxt = last.argmax(-1, keepdim=True)
    # MTP 预测 B
    mtp_id = None
    if eng.mtp is not None:
        pos = torch.tensor([[cache.get_seq_length() - 1]], device=vs.DEVICE)
        mtp_logits, _ = eng.mtp(nxt, h, pos, eng.mtp_cache)
        mtp_id = mtp_logits[0, -1, :].argmax().item()
    if mtp_id is not None and mtp_id != 1:  # eos=1
        B = torch.tensor([[mtp_id]], device=vs.DEVICE)
        combined = torch.cat([nxt, B], dim=-1)
        logits, cache, h = eng.forward(combined, past_key_values=cache, return_hidden=True)
        n_gen += 2
    else:
        logits, cache, h = eng.forward(nxt, past_key_values=cache, return_hidden=True)
        n_gen += 1
    # 下一轮 A 从 logits 取 (generate 里是循环开头, 这里简化为迭代内取)
t_total = time.time() - t0

print('\n===== 剖析结果 (纯 decode, 不含 prefill) =====', flush=True)
print(f'6轮生成 {n_gen} token, decode 总耗时 {t_total:.1f}s → **{t_total/n_gen:.2f} s/tok**', flush=True)
print(f'decode 前向次数: {C["fwds"]} (均 {t_total/max(C["fwds"],1):.2f}s/forward)', flush=True)
print(f'层 forward 总: {T["layer_total"]:.1f}s ({C["layers"]} 层, 均 {T["layer_total"]/max(C["layers"],1):.3f}s/层)', flush=True)
print(f'  其中 MoE 路由: {T["moe_router"]:.1f}s (均 {T["moe_router"]/max(C["moe_calls"],1)*1000:.0f}ms/层)', flush=True)
print(f'      专家读取等待: {T["moe_read_wait"]:.1f}s (均 {T["moe_read_wait"]/max(C["moe_calls"],1)*1000:.0f}ms/层) ← 主线程等磁盘/缓存', flush=True)
print(f'      专家GEMM: {T["moe_gemm"]:.1f}s (均 {T["moe_gemm"]/max(C["moe_calls"],1)*1000:.0f}ms/层)', flush=True)
print(f'      共享专家: {T["moe_shared"]:.1f}s', flush=True)
attn = T["layer_total"] - (T["moe_router"]+T["moe_read_wait"]+T["moe_gemm"]+T["moe_shared"])
print(f'      注意力≈ {attn:.1f}s (层总-莫埃, 均 {attn/max(C["layers"],1)*1000:.0f}ms/层)', flush=True)
print(f'层加载 H2D (build_pool): {T["build_h2d"]:.1f}s ({C["builds"]} 次, 与层计算重叠)', flush=True)
print(f'MTP 模块 forward: {T["mtp_fwd"]:.1f}s ({C["mtp_calls"]} 次, 均 {T["mtp_fwd"]/max(C["mtp_calls"],1)*1000:.0f}ms/次)', flush=True)
print(f'专家缓存命中: GPU {HIT["gpu_hit"]} / RAM {HIT["ram_hit"]} / 磁盘miss {HIT["disk_miss"]}', flush=True)
print(f'  (命中率 {100*(HIT["gpu_hit"]+HIT["ram_hit"])/max(HIT["gpu_hit"]+HIT["ram_hit"]+HIT["disk_miss"],1):.1f}%)', flush=True)
print(f'磁盘读线程耗时(总): {T["lru_disk"]:.1f}s / 专家读取调用 {T["get_expert"]:.1f}s', flush=True)
print('===== DONE =====', flush=True)
