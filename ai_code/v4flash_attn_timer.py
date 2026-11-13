# -*- coding: utf-8 -*-
"""V4Flash 组件计时 — 拆分 attention vs MoE vs 其他, 定位 ot 3s 大头
monkey-patch DeepseekV4Attention.forward 和 V4LazyMoE.forward 计时
"""
import sys, time, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from transformers import AutoTokenizer

T = collections.defaultdict(float)
C = collections.defaultdict(int)

_orig_attn = DeepseekV4Attention.forward
def _timed_attn(self, *a, **k):
    t0 = time.time()
    r = _orig_attn(self, *a, **k)
    T['attn'] += time.time() - t0
    C['attn'] += 1
    return r
DeepseekV4Attention.forward = _timed_attn

_orig_lazy = vs.V4LazyMoE.forward
def _timed_lazy(self, *a, **k):
    t0 = time.time()
    r = _orig_lazy(self, *a, **k)
    T['moe'] += time.time() - t0
    C['moe'] += 1
    return r
vs.V4LazyMoE.forward = _timed_lazy

_orig_fwd = vs.V4FlashEngine.forward
def _timed_fwd(self, *a, **k):
    t0 = time.time()
    r = _orig_fwd(self, *a, **k)
    T['fwd'] += time.time() - t0
    C['fwd'] += 1
    return r
vs.V4FlashEngine.forward = _timed_fwd

print('[at] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)
eng.forward(ids, return_hidden=True)  # 预热

# prefill 一次, 连续 decode 3 token (纯 decode 计时, 不含 prefill)
logits, cache, h = eng.forward(ids, return_hidden=True)
for n in range(3):
    for k in T: T[k] = 0.0
    for k in C: C[k] = 0
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    fwd = T['fwd']
    attn = T['attn']
    moe = T['moe']
    other = max(0, fwd - attn - moe)
    print(f'  token{n+1}: fwd={fwd:.3f}s | attn={attn:.3f}s({attn/fwd*100:.0f}%) | '
          f'moe={moe:.3f}s({moe/fwd*100:.0f}%) | other={other:.3f}s({other/fwd*100:.0f}%)', flush=True)
print('[at] DONE', flush=True)
