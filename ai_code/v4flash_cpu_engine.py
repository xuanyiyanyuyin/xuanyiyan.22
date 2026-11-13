# -*- coding: utf-8 -*-
"""引擎 decode 实际 CPU 用核 — 回答"20线程能不能全用"
"""
import sys, time, ctypes
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class FT(ctypes.Structure):
    _fields_ = [('dwLow', ctypes.c_ulong), ('dwHigh', ctypes.c_ulong)]
K = ctypes.WinDLL('kernel32.dll')
K.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(FT), ctypes.POINTER(FT),
                              ctypes.POINTER(FT), ctypes.POINTER(FT)]
def cpu_sec():
    ct, et, kt, ut = FT(), FT(), FT(), FT()
    K.GetProcessTimes(ctypes.windll.kernel32.GetCurrentProcess(),
                      ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
    return (kt.dwHigh*2**32 + kt.dwLow + ut.dwHigh*2**32 + ut.dwLow) / 1e7

print('[ce] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)  # 预热

c0 = cpu_sec(); t0 = time.time()
for _ in range(5):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
c1 = cpu_sec(); t1 = time.time()
print(f'[ce] decode5: wall={(t1-t0)/5:.2f}s/tok | CPU时间={c1-c0:.2f}s | 用核={(c1-c0)/(t1-t0):.1f}/20', flush=True)
print('[ce] DONE', flush=True)
