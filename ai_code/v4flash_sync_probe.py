# -*- coding: utf-8 -*-
"""V4Flash sync 拆解探针 — 定位 sync 1.3s 构成
窗口循环 5 段: wait_fut(等上传线程提交) / wait_stream(等 DMA) / pin(pin_memory) / fwd(层计算) / d2h(回 CPU)
用法:
  python v4flash_sync_probe.py            # 默认 pinned 模式
  python v4flash_sync_probe.py --no-pin   # 关闭 pin_memory, 对比 sync 来源
"""
import sys, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--no-pin', action='store_true', help='关闭 pin_memory (no-op _pin_window)')
args = ap.parse_args()

if args.no_pin:
    _orig_pin = vs.V4FlashEngine._pin_window
    vs.V4FlashEngine._pin_window = lambda self, g, ge: None
    print('[probe] --no-pin: _pin_window 已禁用', flush=True)

print('[probe] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
eng._probe_sync = True
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)

# pin 生效检查: 加载后 L0 参数是否已 pinned (若 no-pin 模式则应为 0)
n_pin = sum(1 for p in eng.layers[0].parameters() if p.is_pinned())
n_all = sum(1 for _ in eng.layers[0].parameters())
print(f'[probe] pin-check L0: {n_pin}/{n_all} params pinned', flush=True)

def _report(tag):
    acc = getattr(eng, '_sync_acc', {})
    wf, ws, pn, fw, d2h = (acc.get(k, 0.0) for k in ('wait_fut', 'wait_stream', 'pin', 'fwd', 'd2h'))
    win = acc.get('win', 0)
    tot = wf + ws + pn + fw + d2h
    if tot <= 0:
        print(f'  {tag}: 无探针数据', flush=True)
        return
    def pct(x): return f'{x:.3f}s({x/tot*100:.0f}%)'
    print(f'  {tag}: sync_total={tot:.3f}s (窗口={win}) | '
          f'wait_fut={pct(wf)} wait_stream={pct(ws)} pin={pct(pn)} fwd={pct(fw)} d2h={pct(d2h)}', flush=True)

print('[probe] prefill (首次 pin + 首窗口上传含冷启动)...', flush=True)
logits, cache, h = eng.forward(ids, return_hidden=True)
_report('prefill')

for n in range(3):
    eng._sync_acc = {}
    t0 = time.time()
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    dt = time.time() - t0
    print(f'[probe] token{n+1}: fwd_total={dt:.3f}s', flush=True)
    _report(f'decode#{n+1}')
print('[probe] DONE', flush=True)
