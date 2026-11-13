# -*- coding: utf-8 -*-
"""V4Flash 线程池搭配扫描 v2 — 只 prefill 一次复用 cache, 每搭配重建池 decode 8 token
快: 每搭配 ~40s, 8种 ~6分钟 (不重新冷上传43层)
"""
import sys, time, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

COMBOS = [(8,2),(8,4),(8,6),(6,4),(6,6),(6,8),(4,8),(4,10),(10,2),(10,4)]
N_TOK = 8

print('[scan] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
prompt = "请写一篇关于人工智能的短文。"
ids = tok(prompt, return_tensors='pt').input_ids[:, :30].to(vs.DEVICE)

def set_pools(exp, bld):
    try: eng.expert_pool.shutdown(wait=False)
    except Exception: pass
    try: eng.build_pool.shutdown(wait=False)
    except Exception: pass
    eng.expert_pool = ThreadPoolExecutor(max_workers=exp)
    eng.build_pool = ThreadPoolExecutor(max_workers=bld)

# 首次 prefill (复用 cache, 每搭配从该 cache 续 decode, 8token增长影响小)
print('[scan] prefill (复用 cache)...', flush=True)
logits, cache, h = eng.forward(ids, return_hidden=True)

results = []
for exp, bld in COMBOS:
    if exp + bld > 14:
        continue
    set_pools(exp, bld)
    gc.collect()
    # 从 cache 续 decode 8 token
    _log, _cache, _h = logits, cache, h
    t0 = time.time()
    ok = True
    for _ in range(N_TOK):
        try:
            last = _log[:, -1, :].clone().float()
            A = last.argmax(-1, keepdim=True)
            _log, _cache, _h = eng.forward(A, past_key_values=_cache, return_hidden=True)
        except Exception as e:
            print(f'  (expert={exp},build={bld}) 中断: {str(e)[:80]}', flush=True)
            ok = False
            break
    dt = time.time() - t0
    if ok:
        per = dt / N_TOK
        results.append((exp, bld, dt, per))
        print(f'  (expert={exp:2d}, build={bld:2d}): {per:.2f} s/tok', flush=True)

print('\n===== 搭配扫描结果 (decode 8tok, 复用cache) =====', flush=True)
results.sort(key=lambda x: x[3])
for exp, bld, dt, per in results:
    mark = '  ← 最优' if per == results[0][3] else ''
    print(f'  expert={exp:2d} build={bld:2d}: {per:.2f} s/tok{mark}', flush=True)
print(f'\n[scan] 最优: expert={results[0][0]} build={results[0][1]} = {results[0][3]:.2f} s/tok', flush=True)
print('[scan] DONE', flush=True)
