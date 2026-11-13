# -*- coding: utf-8 -*-
"""Page Faults 深度探针 v2 — decode 期间每 0.5s 采样工作集/页文件/系统可用内存
确认: 是否换页(硬缺页) + 工作集是否被 trim + 系统内存是否耗尽
"""
import sys, time, ctypes, threading
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class FT(ctypes.Structure):
    _fields_ = [('dwLowDateTime', ctypes.c_ulong), ('dwHighDateTime', ctypes.c_ulong)]
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

_K32 = ctypes.WinDLL('kernel32.dll'); _PSAPI = ctypes.WinDLL('psapi.dll')
_PSAPI.GetPerformanceInfo.argtypes = [ctypes.POINTER(PI), ctypes.c_ulong]
_PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]

def mem_snap():
    h = ctypes.windll.kernel32.GetCurrentProcess()
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    _PSAPI.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    pi = PI(); pi.cb = ctypes.sizeof(PI)
    _PSAPI.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return (pmc.WorkingSetSize/1e9, pmc.PagefileUsage/1e9, pmc.PageFaultCount,
            pi.PhysicalTotal*pi.PageSize/1e9, pi.PhysicalAvailable*pi.PageSize/1e9, pi.CommitTotal*pi.PageSize/1e9)

print('[pf2] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

# 空闲基线
ws0, pf0, f0, pt0, av0, ct0 = mem_snap()
print(f'[pf2] 空闲基线: WorkingSet={ws0:.1f}GB Pagefile={pf0:.1f}GB | 系统物理={pt0:.0f}GB 可用={av0:.1f}GB Commit={ct0:.0f}GB', flush=True)

# decode 期间采样
SAMPLES = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        SAMPLES.append(mem_snap())
        time.sleep(0.5)
th = threading.Thread(target=sampler, daemon=True); th.start()

t0 = time.time()
for _ in range(5):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
dt = time.time() - t0
stop.set(); th.join()

print(f'[pf2] decode 5 token: {dt:.1f}s ({dt/5:.2f}s/tok)', flush=True)
if SAMPLES:
    ws_all = [s[0] for s in SAMPLES]; pf_all = [s[1] for s in SAMPLES]
    av_all = [s[4] for s in SAMPLES]; f_all = [s[2] for s in SAMPLES]
    print(f'[pf2] WorkingSet: {min(ws_all):.1f}→{max(ws_all):.1f}GB (avg {sum(ws_all)/len(ws_all):.1f})', flush=True)
    print(f'[pf2] Pagefile: {min(pf_all):.1f}→{max(pf_all):.1f}GB (增长{max(pf_all)-min(pf_all):.1f}GB)', flush=True)
    print(f'[pf2] 系统可用内存: {min(av_all):.1f}→{max(av_all):.1f}GB (avg {sum(av_all)/len(av_all):.1f})', flush=True)
    pf_rate = (max(f_all) - min(f_all)) / dt
    print(f'[pf2] page_faults/sec ≈ {pf_rate:,.0f}', flush=True)
print('[pf2] DONE', flush=True)
