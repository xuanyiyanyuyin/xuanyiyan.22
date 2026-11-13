# -*- coding: utf-8 -*-
"""Page Faults + CPU 利用率探针 — 空闲 vs decode 稳态对比
验证: Windows 虚拟内存管理器在 LRU+pinned+build 压力下是否频繁 page fault
指标: page_faults/sec | cpu 用核数 | WorkingSet | Pagefile
"""
import sys, time, ctypes
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

# 显式 WinDLL + argtypes (ctypes.windll 默认 int 参数会破坏 64 位指针)
_K32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
_PSAPI = ctypes.WinDLL('psapi.dll', use_last_error=True)
_K32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT)]
_K32.GetProcessTimes.restype = ctypes.c_int
_PSAPI.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
_PSAPI.GetProcessMemoryInfo.restype = ctypes.c_int

def snap():
    """返回 (cpu_sec, page_faults, working_set_gb, pagefile_gb)"""
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ct, et, kt, ut = FT(), FT(), FT(), FT()
    _K32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
    def to_sec(t): return (t.dwHighDateTime * 2**32 + t.dwLowDateTime) / 1e7
    cpu = to_sec(kt) + to_sec(ut)
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    _PSAPI.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return cpu, pmc.PageFaultCount, pmc.WorkingSetSize/1e9, pmc.PagefileUsage/1e9

def measure(tag, seconds):
    c0, f0, ws0, pf0 = snap()
    t0 = time.time()
    time.sleep(seconds)
    c1, f1, ws1, pf1 = snap()
    wall = time.time() - t0
    print(f'[pf][{tag:10s}] wall={wall:.1f}s | '
          f'page_faults/sec={(f1-f0)/wall:,.0f} | '
          f'cpu核数={(c1-c0)/wall:.1f}/20 | '
          f'WorkingSet={ws0:.1f}→{ws1:.1f}GB | Pagefile={pf0:.1f}→{pf1:.1f}GB', flush=True)

print('[pf] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)  # 预热

print('[pf] 空闲 5s (引擎驻留, 不 decode)...', flush=True)
measure('idle', 5)

print('[pf] decode 3 token (测运行期 page faults)...', flush=True)
f0 = snap()[1]; t0 = time.time()
for _ in range(3):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
dt = time.time() - t0
f1 = snap()[1]
print(f'[pf][decode] wall={dt:.1f}s ({dt/3:.1f}s/tok) | page_faults/sec={(f1-f0)/dt:,.0f}', flush=True)
print('[pf] DONE', flush=True)
