# -*- coding: utf-8 -*-
"""build_pool 线程数扫描 — 同步 vs 2/4/6/8 线程, 配合 expert_pool
no-build 对照发现: 停 build 池 decode 3.39→2.45s (-28%), build线程是净负收益?
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[bs] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def _sync_submit(fn, *a, **k):
    fn(*a, **k)
    f = Future(); f.set_result(None); return f

def set_pools(build_mode, expert, d2h=4):
    try: eng.expert_pool.shutdown(wait=True)
    except Exception: pass
    try: eng.build_pool.shutdown(wait=True)
    except Exception: pass
    try: eng.d2h_pool.shutdown(wait=True)
    except Exception: pass
    gc.collect()
    eng.expert_pool = ThreadPoolExecutor(max_workers=expert)
    eng.d2h_pool = ThreadPoolExecutor(max_workers=d2h)
    if build_mode == 'sync':
        eng.build_pool = ThreadPoolExecutor(max_workers=1)
        eng.build_pool.submit = _sync_submit   # 同步执行 (不占线程)
    else:
        eng.build_pool = ThreadPoolExecutor(max_workers=build_mode)

def run(n=3):
    _l, _c, _h = logits, cache, h
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    return (time.time() - t0) / n

# 组合: (build_mode, expert)
COMBOS = [('sync', 6), (2, 6), (4, 6), (6, 6), (8, 6), (2, 8), (4, 8), ('sync', 8)]
results = []
for bm, ex in COMBOS:
    try:
        set_pools(bm, ex)
        per = run()
        results.append((bm, ex, per))
        print(f'[bs] build={bm} expert={ex}: {per:.2f} s/tok', flush=True)
    except Exception as e:
        print(f'[bs] build={bm} expert={ex}: FAIL {str(e)[:60]}', flush=True)

print('\n[bs] ===== 排序 =====', flush=True)
results.sort(key=lambda x: x[2])
for bm, ex, per in results:
    mark = '  ← 最优' if per == results[0][2] else ''
    print(f'[bs] build={bm:4} expert={ex}: {per:.2f} s/tok{mark}', flush=True)
print('[bs] DONE', flush=True)
