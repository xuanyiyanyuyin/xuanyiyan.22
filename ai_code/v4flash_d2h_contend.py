# -*- coding: utf-8 -*-
"""D2H 抢占验证 — 对照: D2H异步(开) vs D2H同步(关), 测 pread 延迟分布 + decode 时间
如果关 D2H 后 pread 变快 → D2H 抢内存带宽是 pread 慢根因
"""
import sys, time, statistics
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import Future
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# pread 延迟采集
PREAD = []
_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    PREAD.append(time.time() - t0)
    return r
vs.ExpertReader.read_expert = _timed_read

def _pct(v, p):
    if not v: return 0
    v = sorted(v)
    return v[min(len(v)-1, int(len(v)*p))] * 1000

print('[ct] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def run(n, tag):
    global PREAD
    PREAD = []
    _l, _c, _h = logits, cache, h
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    if PREAD:
        print(f'[ct][{tag}] decode{n}={dt/n:.2f}s/tok | pread n={len(PREAD)} '
              f'p50={_pct(PREAD,.5):.0f}ms p90={_pct(PREAD,.9):.0f}ms p99={_pct(PREAD,.99):.0f}ms '
              f'max={max(PREAD)*1000:.0f}ms avg={sum(PREAD)/len(PREAD)*1000:.0f}ms', flush=True)
    else:
        print(f'[ct][{tag}] decode{n}={dt/n:.2f}s/tok (无 pread)', flush=True)

# D2H 异步 (正常)
run(3, 'D2H-ASYNC')

# D2H 同步 (d2h_pool.submit → 同步执行)
_orig_submit = eng.d2h_pool.submit
def _sync_submit(fn, *a, **k):
    fn(*a, **k)
    f = Future(); f.set_result(None); return f
eng.d2h_pool.submit = _sync_submit
run(3, 'D2H-SYNC')
eng.d2h_pool.submit = _orig_submit
print('[ct] DONE', flush=True)
