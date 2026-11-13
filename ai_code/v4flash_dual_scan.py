# -*- coding: utf-8 -*-
"""双盘下线程搭配扫描 — expert/build/d2h 三维, 预热后稳态
14物理核(6P+8E)约束, 找双盘下的最优搭配
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[ds] 加载引擎 (双盘)...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def set_pools(exp, bld, d2h):
    for pool in (eng.expert_pool, eng.build_pool, eng.d2h_pool):
        try: pool.shutdown(wait=True)
        except Exception: pass
    gc.collect()
    eng.expert_pool = ThreadPoolExecutor(max_workers=exp)
    eng.build_pool = ThreadPoolExecutor(max_workers=bld)
    eng.d2h_pool = ThreadPoolExecutor(max_workers=d2h)

def run(warm, n):
    _l, _c, _h = logits, cache, h
    for _ in range(warm):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    times = []
    for _ in range(n):
        t0 = time.time()
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
        times.append(time.time() - t0)
    return sum(times[1:]) / len(times[1:])  # 去首

COMBOS = [
    (6, 6, 4), (8, 6, 4), (10, 4, 4), (12, 2, 4),
    (8, 4, 4), (8, 8, 4), (8, 6, 6), (10, 6, 4),
    (6, 8, 4), (12, 4, 4), (8, 4, 6), (6, 4, 4),
]
results = []
for exp, bld, d2h in COMBOS:
    try:
        set_pools(exp, bld, d2h)
        per = run(warm=3, n=5)
        results.append((exp, bld, d2h, per))
        print(f'[ds] e{exp:2d} b{bld:2d} d{d2h}: {per:.2f} s/tok', flush=True)
    except Exception as e:
        print(f'[ds] e{exp} b{bld} d{d2h}: FAIL {str(e)[:60]}', flush=True)

print('\n[ds] ===== 排序 =====', flush=True)
results.sort(key=lambda x: x[3])
for exp, bld, d2h, per in results:
    mark = '  ← 最优' if per == results[0][3] else ''
    print(f'[ds] expert={exp:2d} build={bld:2d} d2h={d2h:2d}: {per:.2f} s/tok{mark}', flush=True)
print('[ds] DONE', flush=True)
