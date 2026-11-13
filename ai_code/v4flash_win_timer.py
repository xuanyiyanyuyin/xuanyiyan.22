# -*- coding: utf-8 -*-
"""V4Flash 窗口拆解 — 每窗口 上传(H2D) / forward / 回传(D2H) / 同步 计时
定位 other 2.8s: H2D(假异步) vs kernel调度 vs 同步
"""
import sys, time, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

T = collections.defaultdict(float)
C = collections.defaultdict(int)

# monkey-patch: 记录窗口循环各阶段 (通过包装 _upload_layer 和 forward)
_orig_upload = vs.V4FlashEngine._upload_layer
def _timed_upload(self, lid):
    t0 = time.time()
    r = _orig_upload(self, lid)
    T['h2d'] += time.time() - t0
    C['h2d'] += 1
    return r
vs.V4FlashEngine._upload_layer = _timed_upload

# 窗口循环里的 layer.cpu() 是 D2H, 通过包装 layer 的 cpu? 复杂, 改统计 forward 时间
_orig_layer_fwd = None
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
_orig_layer_fwd = DeepseekV4DecoderLayer.forward
def _timed_layer_fwd(self, *a, **k):
    t0 = time.time()
    r = _orig_layer_fwd(self, *a, **k)
    T['layer_fwd'] += time.time() - t0
    C['layer_fwd'] += 1
    return r
DeepseekV4DecoderLayer.forward = _timed_layer_fwd

_orig_fwd = vs.V4FlashEngine.forward
def _timed_fwd(self, *a, **k):
    t0 = time.time()
    r = _orig_fwd(self, *a, **k)
    T['fwd'] += time.time() - t0
    C['fwd'] += 1
    return r
vs.V4FlashEngine.forward = _timed_fwd

print('[wt] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)
eng.forward(ids, return_hidden=True)  # 预热

logits, cache, h = eng.forward(ids, return_hidden=True)
for n in range(2):
    for k in T: T[k] = 0.0
    for k in C: C[k] = 0
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    fwd = T['fwd']
    h2d = T['h2d']
    layer = T['layer_fwd']
    d2h_est = 0  # 每窗口 layer.cpu() 未单独计时
    sync = max(0, fwd - h2d - layer)
    print(f'  token{n+1}: fwd={fwd:.3f}s | H2D={h2d:.3f}s({h2d/fwd*100:.0f}%) | '
          f'layer_fwd={layer:.3f}s({layer/fwd*100:.0f}%) | sync/other={sync:.3f}s({sync/fwd*100:.0f}%)', flush=True)
print('[wt] DONE', flush=True)
