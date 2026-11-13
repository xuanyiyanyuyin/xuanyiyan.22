# -*- coding: utf-8 -*-
"""D2H 异步化内存验证 — 引擎加载 + 2 decode 后检查 RSS / 显存 / pinned 情况"""
import sys, time, ctypes
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

def _mem():
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize/1e9, p.PagefileUsage/1e9, p.PeakWorkingSetSize/1e9

print('[mem] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)
rss, pf, peak = _mem()
print(f'[mem] 加载后: RSS={rss:.1f}GB PageFile={pf:.1f}GB Peak={peak:.1f}GB', flush=True)
print(f'[mem] CUDA alloc={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB', flush=True)

# 预热 + 2 decode (触发 D2H 创建 pinned mirror)
logits, cache, h = eng.forward(ids, return_hidden=True)
for n in range(2):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
rss, pf, peak = _mem()
print(f'[mem] 2 decode 后: RSS={rss:.1f}GB PageFile={pf:.1f}GB Peak={peak:.1f}GB', flush=True)
print(f'[mem] CUDA alloc={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB', flush=True)

# pinned 统计
n_pin = sum(1 for l in eng.layers for p in l.parameters() if getattr(p, '_cpu_pinned', None) is not None)
n_all = sum(1 for l in eng.layers for p in l.parameters())
print(f'[mem] pinned mirror: {n_pin}/{n_all} params', flush=True)
print('[mem] DONE', flush=True)
