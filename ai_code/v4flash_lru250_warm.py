# -*- coding: utf-8 -*-
"""LRU 250 vs 1200 热页缓存验证 — 预热 12 token 让页缓存充分热, 再对比稳态
回答: LRU=250 的快(2.99s)是否依赖页缓存热? 长对话下是否持续?
"""
import sys, time, ctypes, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
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

print('[w] 加载引擎 (LRU=250)...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def run_lru(lru, warm, n):
    global PREAD
    eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=lru, device=vs.DEVICE)
    gc.collect()
    _l, _c, _h = logits, cache, h
    # 预热 (填 LRU + 页缓存)
    for _ in range(warm):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    # 稳态
    PREAD = []
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    pa = sum(PREAD)/len(PREAD)*1000 if PREAD else 0
    print(f'[w] LRU={lru} 预热{warm} 稳态{n}: {dt/n:.2f}s/tok | pread avg={pa:.1f}ms n={len(PREAD)}', flush=True)

# 先用 LRU=250 预热 12 token (页缓存热), 再对比
print('[w] --- 预热阶段 (LRU=250, 12 token) ---', flush=True)
eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=250, device=vs.DEVICE)
gc.collect()
_l, _c, _h = logits, cache, h
for i in range(12):
    A = _l[:, -1, :].argmax(-1, keepdim=True)
    _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    if i % 3 == 2:
        print(f'[w]   预热 #{i+1} done', flush=True)
logits, cache, h = _l, _c, _h

# 现在页缓存热了, 对比 250 vs 1200
run_lru(250, 0, 6)
run_lru(1200, 0, 6)
run_lru(250, 0, 6)
print('[w] DONE', flush=True)
