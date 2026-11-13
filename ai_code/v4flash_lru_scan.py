# -*- coding: utf-8 -*-
"""LRU 容量扫描 — 内存压力 vs 命中率权衡
当前 lru=1200(14.4GB) + 层pinned 13GB = 27GB 挤爆 32GB → 换页
测 lru=600/800/1000/1200 的 decode 速度 + 系统可用内存 + faults
"""
import sys, time, ctypes, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class PI(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('CommitTotal', ctypes.c_size_t), ('CommitLimit', ctypes.c_size_t),
                ('CommitPeak', ctypes.c_size_t), ('PhysicalTotal', ctypes.c_size_t),
                ('PhysicalAvailable', ctypes.c_size_t), ('SystemCache', ctypes.c_size_t),
                ('KernelTotal', ctypes.c_size_t), ('KernelPaged', ctypes.c_size_t),
                ('KernelNonpaged', ctypes.c_size_t), ('PageSize', ctypes.c_size_t),
                ('HandleCount', ctypes.c_ulong), ('ProcessCount', ctypes.c_ulong), ('ThreadCount', ctypes.c_ulong)]
_PSAPI = ctypes.WinDLL('psapi.dll')
_PSAPI.GetPerformanceInfo.argtypes = [ctypes.POINTER(PI), ctypes.c_ulong]
def avail_gb():
    pi = PI(); pi.cb = ctypes.sizeof(PI)
    _PSAPI.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return pi.PhysicalAvailable * pi.PageSize / 1e9

print('[lru] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def run_with_lru(lru):
    # 重建 LRU (空缓存), 改容量
    eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=lru, device=vs.DEVICE)
    gc.collect()
    # 预热 1 token (填 LRU)
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits2, cache2, h2 = eng.forward(A, past_key_values=cache, return_hidden=True)
    # 稳态 3 token
    av0 = avail_gb()
    t0 = time.time()
    for _ in range(3):
        A = logits2[:, -1, :].argmax(-1, keepdim=True)
        logits2, cache2, h2 = eng.forward(A, past_key_values=cache2, return_hidden=True)
    dt = time.time() - t0
    av1 = avail_gb()
    return dt/3, av0, av1, (eng.expert_lru.hits[0], eng.expert_lru.miss[0])

results = []
for lru in (600, 800, 1000, 1200):
    per, av0, av1, hm = run_with_lru(lru)
    hit = hm[0] / (hm[0] + hm[1]) * 100 if hm[0] + hm[1] else 0
    results.append((lru, per, av0, av1, hit))
    print(f'[lru] lru={lru}: {per:.2f} s/tok | 可用内存 {av0:.1f}→{av1:.1f}GB | LRU命中率={hit:.0f}%', flush=True)

print('\n[lru] ===== 排序 =====', flush=True)
results.sort(key=lambda x: x[1])
for lru, per, av0, av1, hit in results:
    mark = '  ← 最优' if per == results[0][1] else ''
    print(f'[lru] lru={lru:4}: {per:.2f} s/tok (可用内存{av1:.1f}GB) 命中{hit:.0f}%{mark}', flush=True)
print('[lru] DONE', flush=True)
