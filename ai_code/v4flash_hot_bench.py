# -*- coding: utf-8 -*-
"""热加载长基准 — 预热 10 token 后跑 100 token, 每 token 打印耗时/pread/内存
观察热加载稳态速度门槛与稳定性
用法: python v4flash_hot_bench.py [--lru N] [--tokens N]
"""
import sys, time, ctypes, statistics, argparse, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

_ap = argparse.ArgumentParser()
_ap.add_argument('--lru', type=int, default=1200)
_ap.add_argument('--tokens', type=int, default=100)
_args = _ap.parse_args()

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

# 本轮 pread 统计
CUR = {'n': 0, 's': 0.0}
_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    CUR['n'] += 1; CUR['s'] += time.time() - t0
    return r
vs.ExpertReader.read_expert = _timed_read

print(f'[hb] 加载引擎 (LRU={_args.lru})...', flush=True)
eng = vs.V4FlashEngine()
# 重建 LRU 到指定值
if _args.lru != 1200:
    eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=_args.lru, device=vs.DEVICE)
    gc.collect()
    print(f'[hb] LRU 已设为 {_args.lru} ({( _args.lru*12.6/1000):.1f}GB)', flush=True)
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

# 预热 10 token (页缓存热)
print('[hb] 预热 10 token (页缓存热)...', flush=True)
_l, _c, _h = logits, cache, h
for i in range(10):
    A = _l[:, -1, :].argmax(-1, keepdim=True)
    _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
logits, cache, h = _l, _c, _h
print(f'[hb] 预热完成, 开始 {_args.tokens} token 热加载基准...', flush=True)

TIMES = []
for i in range(_args.tokens):
    CUR['n'] = 0; CUR['s'] = 0.0
    t0 = time.time()
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    dt = time.time() - t0
    TIMES.append(dt)
    pa = CUR['s']/CUR['n']*1000 if CUR['n'] else 0
    av = avail_gb()
    mark = '  ← 慢' if dt > 3.0 else ''
    print(f'[hb] #{i+1:3d}: {dt:.2f}s | pread={CUR["n"]}次 avg={pa:.0f}ms | 可用={av:.1f}GB{mark}', flush=True)

# 汇总
t = TIMES
print(f'\n[hb] === {_args.tokens} token 汇总 (LRU={_args.lru}) ===', flush=True)
print(f'[hb] 平均={statistics.mean(t):.2f}s | 中位={statistics.median(t):.2f}s | min={min(t):.2f} max={max(t):.2f}', flush=True)
t_sorted = sorted(t)
print(f'[hb] p10={t_sorted[9]:.2f} p50={t_sorted[49]:.2f} p90={t_sorted[89]:.2f} p95={t_sorted[94]:.2f}', flush=True)
# 分段统计 (前50 vs 后50)
print(f'[hb] 前50token均={statistics.mean(t[:50]):.2f}s | 后50token均={statistics.mean(t[50:]):.2f}s', flush=True)
print(f'[hb] >3s的token: {sum(1 for x in t if x>3)} 个 | >2.5s: {sum(1 for x in t if x>2.5)} 个', flush=True)
print('[hb] DONE', flush=True)
