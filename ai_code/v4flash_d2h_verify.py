# -*- coding: utf-8 -*-
"""D2H 异步化正确性验证 — 贪心生成 8 token, 检查输出是否连贯 (无数据污染)
同时监控 RSS 内存, 确认 pinned 13GB 没爆
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

def _rss_gb():
    import ctypes
    class P(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('cbNonPaged', ctypes.c_ulong), ('cbPaged', ctypes.c_ulong),
                    ('cbPageFault', ctypes.c_ulong), ('cbNonPagedPool', ctypes.c_ulong), ('cbPagedPool', ctypes.c_ulong),
                    ('cbPageFile', ctypes.c_ulong), ('cbQuotaNonPaged', ctypes.c_ulong), ('cbQuotaPaged', ctypes.c_ulong),
                    ('cbWorkingSet', ctypes.c_size_t)]
    p = P()
    p.cb = ctypes.sizeof(P)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.cbWorkingSet / 1e9, p.cbPageFile / 1e9

print('[verify] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
rss, pf = _rss_gb()
print(f'[verify] 引擎加载 {time.time()-t0:.0f}s | RSS={rss:.1f}GB PageFile={pf:.1f}GB', flush=True)

# 贪心生成 (无采样随机性), 短输出
prompt = '你好，请介绍一下你自己'
ids = tok(prompt, return_tensors='pt').input_ids.to(vs.DEVICE)
t0 = time.time()
out = eng.generate(ids, max_new_tokens=8, do_sample=False)
dt = time.time() - t0
rss, pf = _rss_gb()
print(f'[verify] 生成 8 token 用时 {dt:.1f}s | RSS={rss:.1f}GB PageFile={pf:.1f}GB', flush=True)

text = tok.decode(out[0], skip_special_tokens=True)
print(f'[verify] 输出: {text!r}', flush=True)
print(f'[verify] 生成部分: {tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)!r}', flush=True)
print(f'[verify] LRU hits={eng.expert_lru.hits[0]} miss={eng.expert_lru.miss[0]}', flush=True)
print('[verify] DONE', flush=True)
