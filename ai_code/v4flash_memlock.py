# -*- coding: utf-8 -*-
"""手写内存调度验证 v2 — SetProcessWorkingSetSize + VirtualLock 对抗换页
阶段: baseline → SetProcessWorkingSetSize(保当前工作集) → +VirtualLock(锁LRU热区)
每阶段独立 try, 输出详细状态
"""
import sys, time, ctypes, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class FT(ctypes.Structure):
    _fields_ = [('dwLow', ctypes.c_ulong), ('dwHigh', ctypes.c_ulong)]
class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
class PI(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('CommitTotal', ctypes.c_size_t), ('CommitLimit', ctypes.c_size_t),
                ('CommitPeak', ctypes.c_size_t), ('PhysicalTotal', ctypes.c_size_t),
                ('PhysicalAvailable', ctypes.c_size_t), ('SystemCache', ctypes.c_size_t),
                ('KernelTotal', ctypes.c_size_t), ('KernelPaged', ctypes.c_size_t),
                ('KernelNonpaged', ctypes.c_size_t), ('PageSize', ctypes.c_size_t),
                ('HandleCount', ctypes.c_ulong), ('ProcessCount', ctypes.c_ulong), ('ThreadCount', ctypes.c_ulong)]
K = ctypes.WinDLL('kernel32.dll'); P = ctypes.WinDLL('psapi.dll')
K.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT)]
P.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
P.GetPerformanceInfo.argtypes = [ctypes.POINTER(PI), ctypes.c_ulong]
def snap():
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ct, et, kt, ut = FT(), FT(), FT(), FT()
    K.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
    cpu = (kt.dwHigh*2**32+kt.dwLow + ut.dwHigh*2**32+ut.dwLow)/1e7
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    P.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    pi = PI(); pi.cb = ctypes.sizeof(PI)
    P.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return cpu, pmc.PageFaultCount, pi.PhysicalAvailable*pi.PageSize/1e9, pmc.WorkingSetSize/1e9

print('[ml] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

def run(tag, n=3):
    c0, f0, av0, ws0 = snap()
    t0 = time.time()
    _l, _c, _h = logits, cache, h
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    c1, f1, av1, ws1 = snap()
    print(f'[ml][{tag:20s}] {dt/n:.2f}s/tok | faults={(f1-f0)/dt:,.0f}/s | 可用{av0:.1f}→{av1:.1f}GB | WS={ws1:.1f}GB | 用核={(c1-c0)/dt:.1f}', flush=True)
    return _l, _c, _h

def virtual_lock_lru(eng, n_exp):
    locked = 0; page = 4096
    with eng.expert_lru.lock:
        items = list(eng.expert_lru.cache.items())[-n_exp:]
    for (lid, eid), tensors in items:
        for t in tensors[:3]:
            addr = t.data_ptr(); sz = t.numel() * t.element_size()
            if addr == 0 or sz == 0: continue
            base = addr & ~(page-1)
            size = ((addr - base + sz + page - 1) // page) * page
            try:
                if ctypes.windll.kernel32.VirtualLock(ctypes.c_void_p(base), size):
                    locked += 1
            except Exception:
                pass
    return locked

# 1 baseline
logits, cache, h = run('baseline')

# 2 SetProcessWorkingSetSize: 保当前工作集 (不强制扩大)
c0, f0, av0, ws0 = snap()
hproc = ctypes.windll.kernel32.GetCurrentProcess()
min_ws = int(ws0 * 1e9)  # 当前工作集
max_ws = int(30 * 1024**3)
r = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(hproc, min_ws, max_ws, 0x1)
print(f'[ml] SetProcessWorkingSetSize(min={ws0:.0f}GB,max=30GB) → ret={r}', flush=True)
time.sleep(1)
logits, cache, h = run('set_workingset')

# 3 VirtualLock 锁 LRU 热区 100 专家
n_lk = virtual_lock_lru(eng, 100)
print(f'[ml] VirtualLock 100专家 → 锁 {n_lk} tensor', flush=True)
logits, cache, h = run('virtual_lock_100')

# 4 VirtualLock 300 专家
n_lk2 = virtual_lock_lru(eng, 200)
print(f'[ml] 再锁 200专家 (累计300) → 锁 {n_lk2} tensor', flush=True)
logits, cache, h = run('virtual_lock_300')
print('[ml] DONE', flush=True)
