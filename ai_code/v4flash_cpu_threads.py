# -*- coding: utf-8 -*-
"""CPU 20线程能力确认 — 分场景测真并行
纯Python计算(GIL) vs CPU matmul(BLAS多线程) vs 磁盘IO(释放GIL) vs 引擎decode
回答: 14核20线程能不能全用? 引擎为什么没用满?
"""
import sys, time, ctypes, threading
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

def p(*a): print(*a, flush=True)
class FT(ctypes.Structure):
    _fields_ = [('dwLowDateTime', ctypes.c_ulong), ('dwHighDateTime', ctypes.c_ulong)]
_K32 = ctypes.WinDLL('kernel32.dll')
_K32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT), ctypes.POINTER(FT)]
def cpu_sec():
    ct, et, kt, ut = FT(), FT(), FT(), FT()
    _K32.GetProcessTimes(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
    return (kt.dwHighDateTime*2**32+kt.dwLowDateTime + ut.dwHighDateTime*2**32+ut.dwLowDateTime) / 1e7

p('=== 0. torch 线程配置 ===')
p(f'torch intra-op={torch.get_num_threads()} inter-op={torch.get_num_interop_threads()}')

p('\n=== 1. CPU matmul (BLAS 多线程, 释放GIL) ===')
a = torch.randn(4096, 4096); b = torch.randn(4096, 4096)
c0 = cpu_sec(); t0 = time.time()
for _ in range(5): c = a @ b
c1 = cpu_sec(); t1 = time.time()
p(f'5次4096@4096: wall={(t1-t0):.2f}s | CPU时间={c1-c0:.2f}s | 用核={(c1-c0)/(t1-t0):.1f}/20')

p('\n=== 2. 磁盘 IO 并行 (pread, 释放GIL) ===')
reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)
import random
lids = random.choices(range(43), k=80); eids = random.choices(range(256), k=80)
from concurrent.futures import ThreadPoolExecutor
for n in (1, 4, 8, 12, 20):
    pool = ThreadPoolExecutor(max_workers=n)
    c0 = cpu_sec(); t0 = time.time()
    futs = [pool.submit(reader.read_expert, lids[i], eids[i]) for i in range(80)]
    for f in futs: f.result()
    c1 = cpu_sec(); t1 = time.time()
    pool.shutdown()
    p(f'{n:2d}线程pread80专家: wall={t1-t0:.2f}s | CPU时间={(c1-c0):.2f}s | 用核={(c1-c0)/(t1-t0):.1f}/20')

p('\n=== 3. 引擎 decode 实际用核 ===')
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)  # 预热
c0 = cpu_sec(); t0 = time.time()
for _ in range(3):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
c1 = cpu_sec(); t1 = time.time()
p(f'引擎decode3token: wall={(t1-t0)/3:.2f}s/tok | CPU时间={(c1-c0):.2f}s | 用核={(c1-c0)/(t1-t0):.1f}/20')
p('=== DONE ===')
