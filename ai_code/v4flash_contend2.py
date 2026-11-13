# -*- coding: utf-8 -*-
"""引擎内竞争源定位 — 分别停 d2h/build 池, 测 pread avg + decode 时间
定位: 哪个池 (D2H内存带宽 / build H2D / 线程竞争) 拖慢磁盘读
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import Future
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

PREAD = []
_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    PREAD.append(time.time() - t0)
    return r
vs.ExpertReader.read_expert = _timed_read

print('[c2] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def _sync_submit(fn, *a, **k):
    fn(*a, **k)
    f = Future(); f.set_result(None); return f

def run(tag, n=2):
    global PREAD
    PREAD = []
    _l, _c, _h = logits, cache, h
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    avg = sum(PREAD)/len(PREAD)*1000 if PREAD else 0
    print(f'[c2][{tag:16s}] decode={dt/n:.2f}s/tok | pread avg={avg:.0f}ms n={len(PREAD)}', flush=True)

# 全开
run('full')

# 停 d2h (D2H 同步)
_os = eng.d2h_pool.submit
eng.d2h_pool.submit = _sync_submit
run('no-d2h')
eng.d2h_pool.submit = _os

# 停 build (H2D 同步)
_os2 = eng.build_pool.submit
eng.build_pool.submit = _sync_submit
run('no-build')
eng.build_pool.submit = _os2

# 停 d2h + build
eng.d2h_pool.submit = _sync_submit
eng.build_pool.submit = _sync_submit
run('no-d2h-build')
eng.d2h_pool.submit = _os
eng.build_pool.submit = _os2
print('[c2] DONE', flush=True)
