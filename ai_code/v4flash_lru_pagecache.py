# -*- coding: utf-8 -*-
"""LRU 缩容 vs 页缓存 — 少放专家, 腾内存给 OS 页缓存, 能否让专家磁盘读命中页缓存变快?
测: 不同 LRU 下 pread avg (页缓存命中应 <5ms, 裸盘 ~9ms) + decode + Standby + 可用内存
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
def sys_mem():
    pi = PI(); pi.cb = ctypes.sizeof(PI)
    _PSAPI.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    avail = pi.PhysicalAvailable * pi.PageSize / 1e9
    cache = pi.SystemCache * pi.PageSize / 1e9   # 系统缓存 (含Standby)
    return avail, cache

# pread 计时
PREAD = []
_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    PREAD.append(time.time() - t0)
    return r
vs.ExpertReader.read_expert = _timed_read

print('[lp] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def run_lru(lru, warm=2, n=3):
    global PREAD
    eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=lru, device=vs.DEVICE)
    gc.collect()
    # 预热 warm token (填 LRU + 让页缓存热)
    _l, _c, _h = logits, cache, h
    for _ in range(warm):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    # 稳态 n token
    PREAD = []
    av0, ca0 = sys_mem()
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    av1, ca1 = sys_mem()
    pa = sum(PREAD)/len(PREAD)*1000 if PREAD else 0
    hitr = eng.expert_lru.hits[0]/(eng.expert_lru.hits[0]+eng.expert_lru.miss[0])*100
    print(f'[lp] LRU={lru:4}: {dt/n:.2f}s/tok | pread avg={pa:.1f}ms n={len(PREAD)} | '
          f'可用{av0:.1f}→{av1:.1f}GB 缓存{ca0:.1f}→{ca1:.1f}GB | LRU命中={hitr:.0f}%', flush=True)

for lru in (1200, 600, 400, 250, 150):
    run_lru(lru)
print('[lp] DONE', flush=True)
