# -*- coding: utf-8 -*-
"""双盘并发 + LRU 缩容实验 — 页缓存腾出空间后, 双盘(C+D)并发读专家是否突破单盘
对比: 不同 LRU 下 decode + pread avg
引擎自动 remap C 盘分片 (C:/ai_v4flash_experts)
"""
import sys, time, ctypes, gc, os
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

print('[dd] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
# 确认 C 盘 remap
n_remap = len([s for s in os.listdir(r'C:/ai_v4flash_experts') if s.endswith('.safetensors')]) if os.path.exists(r'C:/ai_v4flash_experts') else 0
print(f'[dd] C盘分片: {n_remap} shards remapped ({"双盘" if n_remap>0 else "单盘"})', flush=True)
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def run_lru(lru, warm=3, n=4):
    global PREAD
    eng.expert_lru = vs.ExpertLRU(eng.expert_reader, max_items=lru, device=vs.DEVICE)
    gc.collect()
    _l, _c, _h = logits, cache, h
    for _ in range(warm):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    PREAD = []
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    pa = sum(PREAD)/len(PREAD)*1000 if PREAD else 0
    print(f'[dd] LRU={lru:4}: {dt/n:.2f}s/tok | pread avg={pa:.1f}ms n={len(PREAD)}', flush=True)

for lru in (1200, 600, 400):
    run_lru(lru)

# 清除 remap 测单盘对比
eng.expert_reader._remaps.clear()
eng.expert_reader._hdr_cache.clear()
print('[dd] 已清除 remap → 单盘 (D盘)', flush=True)
run_lru(1200)
print('[dd] DONE', flush=True)
