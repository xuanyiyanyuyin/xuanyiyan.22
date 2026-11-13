# -*- coding: utf-8 -*-
"""专家读取全链路详细探针 v1 — 拆解 decode 中专家读取的每一环
覆盖:
  ① LRU 三层命中分布 (GPU热层 / RAM LRU / 磁盘读)
  ② read_expert 6-tensor 串行读耗时 (按 w1/w2/w3/s1/s2/s3 细分)
  ③ f.read 按大小档耗时 (4.2MB vs 0.25MB, 看固定开销) + clone
  ④ 每层 MoE 耗时分布 (43层, 找慢层)
  ⑤ MTP 专家独立路径 (get_mtp_expert)
"""
import sys, time, collections, struct, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# ─── 统计容器 ───
S = {
    'gpu_hit': 0, 'ram_hit': 0, 'disk_read': 0,      # 专家三层
    'layer_miss': collections.defaultdict(int),       # 每层磁盘 miss
    'get_exp_n': 0, 'get_exp_s': 0.0,                 # get_expert 总
    'mtp_n': 0, 'mtp_s': 0.0,                         # MTP 专家
    'moe_layer': collections.defaultdict(float),       # 每层 MoE 秒
    'moe_n': 0, 'moe_total': 0.0,
    'read_exp_n': 0, 'read_exp_s': 0.0,               # read_expert
    'tensor': collections.defaultdict(lambda: [0.0, 0]),   # 按 tensor 名
    'fread_sz': collections.defaultdict(lambda: [0.0, 0]), # f.read 按大小档
    'clone_sz': collections.defaultdict(lambda: [0.0, 0]), # clone 按大小档
    'fread_max': 0.0, 'clone_max': 0.0,
}

# ─── patch ExpertReader._read: f.read/clone 按大小分类 ───
_orig_read = vs.ExpertReader._read
def _timed_read(self, key):
    shard = self.disk[key]
    f = self._fh(shard)
    hd = self._hdr_cache.get(shard)
    if hd is None:
        with open(self._shard_path(shard), 'rb') as tf:
            n = struct.unpack('<Q', tf.read(8))[0]
            hd = (n + 8, json.loads(tf.read(n)))
        self._hdr_cache[shard] = hd
    base, hdr = hd
    m = hdr[key]
    o0, o1 = m['data_offsets']
    ln = o1 - o0
    t0 = time.time()
    f.seek(base + o0); data = f.read(ln)
    dt = time.time() - t0
    S['fread_sz'][('W' if ln > 1e6 else 'S')][0] += dt
    S['fread_sz'][('W' if ln > 1e6 else 'S')][1] += 1
    if dt > S['fread_max']: S['fread_max'] = dt
    t0 = time.time()
    r = torch.frombuffer(data, dtype=vs.DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone()
    dt = time.time() - t0
    S['clone_sz'][('W' if ln > 1e6 else 'S')][0] += dt
    S['clone_sz'][('W' if ln > 1e6 else 'S')][1] += 1
    if dt > S['clone_max']: S['clone_max'] = dt
    return r
vs.ExpertReader._read = _timed_read

# ─── patch ExpertReader.read_expert: 每 tensor 细分 ───
_orig_read_exp = vs.ExpertReader.read_expert
def _timed_read_exp(self, lid, eid):
    t0 = time.time()
    r = _orig_read_exp(self, lid, eid)
    dt = time.time() - t0
    S['read_exp_n'] += 1; S['read_exp_s'] += dt
    return r
vs.ExpertReader.read_expert = _timed_read_exp

# ─── patch ExpertLRU.get: RAM命中 vs 磁盘 ───
_orig_lru = vs.ExpertLRU.get
def _timed_lru(self, lid, eid):
    key = (lid, eid)
    with self.lock:
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            self.hits[0] += 1
            S['ram_hit'] += 1
            return hit
    t = self.reader.read_expert(lid, eid)
    S['disk_read'] += 1
    S['layer_miss'][lid] += 1          # 每层磁盘 miss
    with self.lock:
        self.miss[0] += 1
        self.cache[key] = t
        while len(self.cache) > self.max:
            self.cache.popitem(last=False)
    return t
vs.ExpertLRU.get = _timed_lru

# ─── patch get_expert: GPU热层分类 ───
_orig_get = vs.V4FlashEngine.get_expert
def _timed_get(self, lid, eid):
    key = (lid, eid)
    t0 = time.time()
    with self.exp_lock:
        gh = self.gpu_experts.get(key)
        if gh is not None:
            self.gpu_experts.move_to_end(key)
            S['gpu_hit'] += 1
            S['get_exp_n'] += 1; S['get_exp_s'] += time.time() - t0
            return gh
    t = self.expert_lru.get(lid, eid)
    g = tuple(x.to(vs.DEVICE, non_blocking=True) for x in t)
    with self.exp_lock:
        self.gpu_experts[key] = g
        while len(self.gpu_experts) > self.gpu_exp_max:
            self.gpu_experts.popitem(last=False)
    S['get_exp_n'] += 1; S['get_exp_s'] += time.time() - t0
    return g
vs.V4FlashEngine.get_expert = _timed_get

# ─── patch get_mtp_expert ───
_orig_mtp = vs.V4FlashEngine.get_mtp_expert
def _timed_mtp(self, eid):
    t0 = time.time()
    r = _orig_mtp(self, eid)
    S['mtp_n'] += 1; S['mtp_s'] += time.time() - t0
    return r
vs.V4FlashEngine.get_mtp_expert = _timed_mtp

# ─── patch V4LazyMoE.forward: 每层 MoE 计时 ───
_orig_moe = vs.V4LazyMoE.forward
def _timed_moe(self, *a, **k):
    t0 = time.time()
    r = _orig_moe(self, *a, **k)
    dt = time.time() - t0
    if self.mtp:
        S['moe_mtp'] = S.get('moe_mtp', 0.0) + dt
        S['mtp_moe_n'] = S.get('mtp_moe_n', 0) + 1
    else:
        S['moe_layer'][self.lid] += dt
        if self.lid == 0:
            S.setdefault('l0_seq', []).append(dt)
        S['moe_n'] += 1; S['moe_total'] += dt
    return r
vs.V4LazyMoE.forward = _timed_moe

# ─── 跑 ───
print('[x] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

for n in range(5):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

# ─── 报告 ───
N = 5
def g(container, key):
    v = container.get(key, [0.0, 0]); return v
print(f'\n=== 专家读取全链路 (decode {N} token, 5轮平均) ===', flush=True)

tot_exp = S['gpu_hit'] + S['ram_hit'] + S['disk_read']
if tot_exp:
    print(f'[x] ① 专家三层命中: GPU热层={S["gpu_hit"]}({S["gpu_hit"]/tot_exp*100:.0f}%) '
          f'RAM_LRU={S["ram_hit"]}({S["ram_hit"]/tot_exp*100:.0f}%) '
          f'磁盘读={S["disk_read"]}({S["disk_read"]/tot_exp*100:.0f}%)  总={tot_exp}', flush=True)
if S['get_exp_n']:
    print(f'[x]    get_expert: avg={S["get_exp_s"]/S["get_exp_n"]*1000:.0f}ms/专家 '
          f'({S["get_exp_s"]/N:.2f}s/token)', flush=True)
if S['read_exp_n']:
    print(f'[x] ② read_expert(磁盘读): n={S["read_exp_n"]} avg={S["read_exp_s"]/S["read_exp_n"]*1000:.1f}ms '
          f'({S["read_exp_s"]/N:.2f}s/token)', flush=True)
if S['tensor'] or S['fread_sz']:
    for tag in ('W', 'S'):
        fr = g(S['fread_sz'], tag); cl = g(S['clone_sz'], tag)
        if fr[1]:
            print(f'[x] ③ {"权重4.2MB" if tag=="W" else "scale0.3MB"}: '
                  f'f.read avg={fr[0]/fr[1]*1000:5.1f}ms n={fr[1]}(累计{N}token) | clone avg={cl[0]/cl[1]*1000:.1f}ms', flush=True)
    print(f'[x]    f.read max={S["fread_max"]*1000:.0f}ms clone max={S["clone_max"]*1000:.0f}ms', flush=True)
if S['moe_n']:
    moes = sorted(S['moe_layer'].items())
    slow = sorted(S['moe_layer'].items(), key=lambda x: -x[1])[:5]
    fast = sorted(S['moe_layer'].items(), key=lambda x: x[1])[:3]
    print(f'[x] ④ MoE 总={S["moe_total"]:.2f}s ({S["moe_total"]/N:.2f}s/token) n={S["moe_n"]} '
          f'avg={S["moe_total"]/S["moe_n"]*1000:.0f}ms/层', flush=True)
    print(f'[x]    最慢5层: ' + ' '.join(f'L{l}={t*1000:.0f}ms' for l, t in slow), flush=True)
    if S.get('l0_seq'):
        print(f'[x]    L0 逐token: ' + ' '.join(f'#{i}={t*1000:.0f}ms' for i, t in enumerate(S['l0_seq'])), flush=True)
    print(f'[x]    最快3层: ' + ' '.join(f'L{l}={t*1000:.0f}ms' for l, t in fast), flush=True)
if S['mtp_n']:
    print(f'[x] ⑤ MTP专家: n={S["mtp_n"]} avg={S["mtp_s"]/S["mtp_n"]*1000:.1f}ms/专家 total={S["mtp_s"]:.2f}s', flush=True)
if S.get('mtp_moe_n'):
    print(f'[x]    MTP-MoE(独立): n={S["mtp_moe_n"]} avg={S["moe_mtp"]/S["mtp_moe_n"]*1000:.0f}ms total={S["moe_mtp"]:.2f}s', flush=True)
if S['layer_miss']:
    lm = sorted(S['layer_miss'].items())
    top_miss = sorted(S['layer_miss'].items(), key=lambda x: -x[1])[:8]
    print(f'[x] ⑥ 每层磁盘miss (top8): ' + ' '.join(f'L{l}={c}' for l, c in top_miss), flush=True)
    print(f'[x]    前5层miss={sum(S["layer_miss"][i] for i in range(5))} '
          f'后5层miss={sum(S["layer_miss"][i] for i in range(38,43))} '
          f'中33层miss={sum(S["layer_miss"][i] for i in range(5,38))}', flush=True)

# 磁盘读总量 (修正: 区分累计 vs per-token)
N = 5
tot_gb = S['disk_read'] * 12.6 / 1000
tok_gb = tot_gb / N
print(f'[x] 磁盘读总量: {S["disk_read"]}专家/累计{tot_gb:.1f}GB | '
      f'{S["disk_read"]/N:.0f}专家/token × 12.6MB = {tok_gb:.2f}GB/token', flush=True)
print(f'[x] 修正: 之前误把累计当per-token(14.7GB), 实际 {tok_gb:.2f}GB/token', flush=True)
print('[x] DONE', flush=True)
