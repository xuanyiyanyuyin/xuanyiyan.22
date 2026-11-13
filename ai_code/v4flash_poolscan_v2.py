# -*- coding: utf-8 -*-
"""现引擎 (D2H异步后) 线程池扫描 — expert_pool / build_pool / d2h_pool 组合
QD 扫描发现: 磁盘 QD8 最优 (空闲 3-7ms/op vs 引擎内 33ms/op) → 重扫现引擎最优池
同时 monkey-patch read_expert 监控引擎内 pread 延迟分布
"""
import sys, time, gc, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 组合 (expert, build, d2h) — 控制总线程, 避免 E-core 饿死
COMBOS = [(4, 10, 4), (8, 10, 4), (12, 10, 4), (8, 8, 4), (8, 8, 8), (6, 8, 4)]
N_TOK = 3

# pread 延迟统计
PR = {'n': 0, 's': 0.0, 'max': 0.0, 'slow_100ms': 0}
_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    t0 = time.time()
    r = _orig_read(self, lid, eid)
    dt = time.time() - t0
    PR['n'] += 1; PR['s'] += dt
    if dt > PR['max']: PR['max'] = dt
    if dt > 0.1: PR['slow_100ms'] += 1
    return r
vs.ExpertReader.read_expert = _timed_read

print('[scan2] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下自己", return_tensors='pt').input_ids.to(vs.DEVICE)

def set_pools(exp, bld, d2h):
    try: eng.expert_pool.shutdown(wait=True)
    except Exception: pass
    try: eng.build_pool.shutdown(wait=True)
    except Exception: pass
    try: eng.d2h_pool.shutdown(wait=True)
    except Exception: pass
    gc.collect()
    eng.expert_pool = ThreadPoolExecutor(max_workers=exp)
    eng.build_pool = ThreadPoolExecutor(max_workers=bld)
    eng.d2h_pool = ThreadPoolExecutor(max_workers=d2h)

print('[scan2] prefill (复用 cache)...', flush=True)
logits, cache, h = eng.forward(ids, return_hidden=True)

results = []
for exp, bld, d2h in COMBOS:
    if exp + bld + d2h > 20:
        continue
    set_pools(exp, bld, d2h)
    _log, _cache, _h = logits, cache, h
    PR['n'] = PR['s'] = PR['max'] = PR['slow_100ms'] = 0
    t0 = time.time()
    ok = True
    for _ in range(N_TOK):
        try:
            A = _log[:, -1, :].argmax(-1, keepdim=True)
            _log, _cache, _h = eng.forward(A, past_key_values=_cache, return_hidden=True)
        except Exception as e:
            print(f'  (e{exp},b{bld},d{d2h}) 中断: {str(e)[:80]}', flush=True)
            ok = False
            break
    dt = time.time() - t0
    if ok:
        per = dt / N_TOK
        pread_avg = PR['s'] / PR['n'] * 1000 if PR['n'] else 0
        results.append((exp, bld, d2h, per))
        print(f'  (e{exp:2d},b{bld:2d},d{d2h}): {per:.2f} s/tok | pread n={PR["n"]} avg={pread_avg:.1f}ms max={PR["max"]*1000:.0f}ms >100ms={PR["slow_100ms"]}', flush=True)

print('\n===== 现引擎线程池扫描 (decode 3tok) =====', flush=True)
results.sort(key=lambda x: x[3])
for exp, bld, d2h, per in results:
    mark = '  ← 最优' if per == results[0][3] else ''
    print(f'  expert={exp:2d} build={bld:2d} d2h={d2h:2d}: {per:.2f} s/tok{mark}', flush=True)
print('[scan2] DONE', flush=True)
