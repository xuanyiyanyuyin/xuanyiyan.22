# -*- coding: utf-8 -*-
"""双盘严格验证 — 预热 10 token 后, 双盘/单盘交替多轮, 排除页缓存/系统噪声
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

print('[dv] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
n_remap = len([s for s in os.listdir(r'C:/ai_v4flash_experts') if s.endswith('.safetensors')]) if os.path.exists(r'C:/ai_v4flash_experts') else 0
print(f'[dv] C盘分片: {n_remap} ({"" if n_remap else "单"}双盘)', flush=True)
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

def set_dual(dual):
    eng.expert_reader._remaps.clear()
    eng.expert_reader._hdr_cache.clear()
    if dual:
        C = r'C:/ai_v4flash_experts'
        for s in os.listdir(C):
            if s.endswith('.safetensors'):
                eng.expert_reader.remap(s, C + '/' + s)

def run(n=6):
    global PREAD
    PREAD = []
    _l, _c, _h = logits, cache, h
    t0 = time.time()
    for _ in range(n):
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
    dt = time.time() - t0
    pa = sum(PREAD)/len(PREAD)*1000 if PREAD else 0
    return dt/n, pa

# 预热 10 token (双盘)
set_dual(True)
_l, _c, _h = logits, cache, h
for i in range(10):
    A = _l[:, -1, :].argmax(-1, keepdim=True)
    _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
logits, cache, h = _l, _c, _h
print('[dv] 预热完成, 开始交替对比...', flush=True)

# 交替 双盘/单盘 各 2 轮
for i in range(2):
    set_dual(True);  per, pa = run(); print(f'[dv] 轮{i+1} 双盘: {per:.2f}s/tok pread={pa:.1f}ms', flush=True)
    set_dual(False); per, pa = run(); print(f'[dv] 轮{i+1} 单盘: {per:.2f}s/tok pread={pa:.1f}ms', flush=True)
print('[dv] DONE', flush=True)
