# -*- coding: utf-8 -*-
"""双盘线程配置交替验证 — 预热10token后, e6b6/e8b4/e8b6 交替多轮
排除顺序/页缓存累积效应, 确认真实最优
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[dv2] 加载引擎 (双盘)...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def set_pools(exp, bld, d2h=4):
    for pool in (eng.expert_pool, eng.build_pool, eng.d2h_pool):
        try: pool.shutdown(wait=True)
        except Exception: pass
    gc.collect()
    eng.expert_pool = ThreadPoolExecutor(max_workers=exp)
    eng.build_pool = ThreadPoolExecutor(max_workers=bld)
    eng.d2h_pool = ThreadPoolExecutor(max_workers=d2h)

def run(n=5):
    _l, _c, _h = logits, cache, h
    times = []
    for _ in range(n):
        t0 = time.time()
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
        times.append(time.time() - t0)
    return sum(times[1:])/len(times[1:])

# 预热 12 token (e8b4, 页缓存热)
set_pools(8, 4)
_l, _c, _h = logits, cache, h
for i in range(12):
    A = _l[:, -1, :].argmax(-1, keepdim=True)
    _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
logits, cache, h = _l, _c, _h
print('[dv2] 预热完成, 交替对比...', flush=True)

CFGS = [(8, 4), (6, 6), (8, 6), (6, 6), (8, 4)]
for exp, bld in CFGS:
    set_pools(exp, bld)
    per = run()
    print(f'[dv2] e{exp} b{bld}: {per:.2f} s/tok', flush=True)
print('[dv2] DONE', flush=True)
