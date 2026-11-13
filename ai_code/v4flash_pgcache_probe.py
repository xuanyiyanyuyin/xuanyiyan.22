# -*- coding: utf-8 -*-
"""V4Flash 页缓存影响受控实验 — 专家读取是否依赖 OS 页缓存?
流程: baseline 2 token → 读 15GB 挤掉 Standby → 再 2 token, 对比 pread 平均耗时 / MoE 耗时
如果挤掉页缓存后 pread 变慢 → 页缓存重要 → D2H 异步化(13GB pinned)有风险
如果不变 → 专家冷读为主 → 页缓存无关紧要 → D2H 异步化可行
"""
import sys, time, os, glob, threading
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# === 计时探针: pread(读专家) 和 MoE forward ===
STATS = {'pread_s': 0.0, 'pread_n': 0, 'moe_s': 0.0}

_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    STATS['pread_s'] += time.time() - t0
    STATS['pread_n'] += 1
    return r
vs.ExpertReader.read_expert = _timed_read

_orig_moe = vs.V4LazyMoE.forward
def _timed_moe(self, *a, **k):
    t0 = time.time()
    r = _orig_moe(self, *a, **k)
    STATS['moe_s'] += time.time() - t0
    return r
vs.V4LazyMoE.forward = _timed_moe

def _standby_gb():
    """读 Windows Standby 列表大小 (页缓存近似)"""
    import subprocess
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
        "(Get-Counter '\\Memory\\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue"],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) / 1e9
    except Exception:
        return -1.0

def _squeeze_pagecache(gb=15):
    """顺序读 D 盘 shard 文件 gb GB, 挤掉 Standby 里的专家页"""
    shards = sorted(glob.glob('D:/ai_v4flash/model-*.safetensors'))
    total = 0
    chunk = 64 * 1024 * 1024
    target = gb * 1024**3
    for sp in shards:
        if total >= target:
            break
        with open(sp, 'rb') as f:
            while total < target:
                d = f.read(chunk)
                if not d:
                    break
                total += len(d)
    return total / 1e9

print('[pg] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，请介绍一下你自己", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)  # prefill 预热 (OS 缓存热专家)

miss0 = eng.expert_lru.miss[0]
hits0 = eng.expert_lru.hits[0]

def run_decode(n, tag):
    global logits, cache, h
    for k in STATS: STATS[k] = 0.0
    m0, h0 = eng.expert_lru.miss[0], eng.expert_lru.hits[0]
    t0 = time.time()
    for i in range(n):
        A = logits[:, -1, :].argmax(-1, keepdim=True)
        logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    wall = time.time() - t0
    np_ = STATS['pread_n']
    avg = STATS['pread_s'] / np_ * 1000 if np_ else 0
    dm, dh = eng.expert_lru.miss[0] - m0, eng.expert_lru.hits[0] - h0
    print(f'[pg][{tag}] decode{n} wall={wall:.2f}s | moe={STATS["moe_s"]:.2f}s | '
          f'pread={STATS["pread_s"]:.2f}s/{np_}次 avg={avg:.2f}ms | LRU +miss={dm} +hit={dh} | '
          f'standby={_standby_gb():.1f}GB', flush=True)

print(f'[pg] 初始: standby={_standby_gb():.1f}GB | LRU 累积 hits={hits0} miss={miss0}', flush=True)
run_decode(2, 'BASELINE')

print('[pg] 挤页缓存: 顺序读 15GB shard...', flush=True)
read_gb = _squeeze_pagecache(15)
print(f'[pg] 挤完 {read_gb:.1f}GB, standby={_standby_gb():.1f}GB', flush=True)

run_decode(2, 'AFTER-SQUEEZE')
print('[pg] DONE', flush=True)
