# -*- coding: utf-8 -*-
"""配置稳定验证 — 各 (expert, build) 组合跑 5 token 求平均, 排除噪声
buildscan 3token 噪声大, 用 5token 稳态确认真实最优
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[cv] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def _sync_submit(fn, *a, **k):
    fn(*a, **k)
    f = Future(); f.set_result(None); return f

def set_pools(expert, build):
    for pool in (eng.expert_pool, eng.build_pool, eng.d2h_pool):
        try: pool.shutdown(wait=True)
        except Exception: pass
    gc.collect()
    eng.expert_pool = ThreadPoolExecutor(max_workers=expert)
    eng.d2h_pool = ThreadPoolExecutor(max_workers=4)
    if build == 'sync':
        eng.build_pool = ThreadPoolExecutor(max_workers=1)
        eng.build_pool.submit = _sync_submit
    else:
        eng.build_pool = ThreadPoolExecutor(max_workers=build)

def run(n=5):
    _l, _c, _h = logits, cache, h
    times = []
    for _ in range(n):
        t0 = time.time()
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
        times.append(time.time() - t0)
    # 去首 token (冷启动), 取后 4 个稳态
    return sum(times[1:]) / len(times[1:]), times

COMBOS = [(6, 8), (8, 8), (8, 4), (8, 'sync'), (6, 4), (4, 8), (6, 6), (8, 6)]
results = []
for ex, bd in COMBOS:
    try:
        set_pools(ex, bd)
        per, ts = run()
        results.append((ex, bd, per))
        print(f'[cv] expert={ex} build={bd}: {per:.2f} s/tok | 各token: ' + ' '.join(f'{t:.1f}' for t in ts), flush=True)
    except Exception as e:
        print(f'[cv] expert={ex} build={bd}: FAIL {str(e)[:60]}', flush=True)

print('\n[cv] ===== 稳态排序 (去首token) =====', flush=True)
results.sort(key=lambda x: x[2])
for ex, bd, per in results:
    mark = '  ← 最优' if per == results[0][2] else ''
    print(f'[cv] expert={ex} build={bd:4}: {per:.2f} s/tok{mark}', flush=True)
print('[cv] DONE', flush=True)
