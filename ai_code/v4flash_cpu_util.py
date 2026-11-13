# -*- coding: utf-8 -*-
"""引擎 CPU 利用率探针 — decode 时进程 CPU 时间 (内核+用户) / wall = 平均用核数
确认引擎是否充分用核 (多线程释放GIL) 还是 CPU 受限
"""
import sys, time, ctypes
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class FT(ctypes.Structure):
    _fields_ = [('dwLowDateTime', ctypes.c_ulong), ('dwHighDateTime', ctypes.c_ulong)]

def cpu_seconds():
    ct, et, kt, ut = FT(), FT(), FT(), FT()
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.kernel32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut))
    if not ok:
        return -1.0
    def to_sec(t): return (t.dwHighDateTime * 2**32 + t.dwLowDateTime) / 1e7
    return to_sec(kt) + to_sec(ut)   # 内核+用户

print('[cu] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

# 预热 1 token
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

# 正式 3 token, 采样 CPU
c0 = cpu_seconds(); w0 = time.time()
for n in range(3):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
w1 = time.time(); c1 = cpu_seconds()

wall = w1 - w0
cpu = c1 - c0
print(f'[cu] decode 3 token: wall={wall:.2f}s ({wall/3:.2f}s/tok)', flush=True)
print(f'[cu] CPU 时间={cpu:.2f}s (内核+用户) | 平均用核数={cpu/wall:.1f} / 20', flush=True)
print(f'[cu] 若用核数≈1-2 → CPU几乎空闲在等IO; 若≈6-10 → CPU繁忙', flush=True)
print('[cu] DONE', flush=True)
