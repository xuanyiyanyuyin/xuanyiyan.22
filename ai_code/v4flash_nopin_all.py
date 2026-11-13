# -*- coding: utf-8 -*-
"""层 pinned 决策实验 — 释放全部层 pinned (pageable) vs 当前 (pinned 13GB)
如果释放后: 可用内存↑ 硬缺页↓ pread快, decode 可能不因 D2H 同步而大幅退化
→ 证明层 pinned(锁页13GB)是负优化, 该回退/改滚动pin
"""
import sys, time, ctypes, subprocess
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 层不 pin 的 offload (pageable mirror)
_orig_offload = vs.V4FlashEngine._offload_window
def _nopin_offload(self, _g, _ge):
    for lid in range(_g, _ge):
        layer = self.layers[lid]
        for name, p in layer.named_parameters():
            if p.is_cuda:
                mir = getattr(p, '_cpu_pinned', None)
                if mir is None:
                    mir = torch.empty_like(p, device='cpu')   # pageable, 不锁页
                    p._cpu_pinned = mir
                mir.copy_(p, non_blocking=False)
                p.data = mir
vs.V4FlashEngine._offload_window = _nopin_offload

def hardfault_ps(seconds=3):
    cmd = ['powershell', '-NoProfile', '-Command',
           f"Get-Counter '\\Memory\\Pages Input/sec' -SampleInterval 1 -MaxSamples {seconds} " +
           "| ForEach-Object { $_.CounterSamples.CookedValue }"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    vals = [float(x) for x in r.stdout.strip().split() if x.replace('.','').replace('-','').isdigit()]
    return sum(vals)/len(vals) if vals else 0

print('[np] 加载引擎 (层不pin)...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

# 确认层 mirror 是否 pin
n_pin = sum(1 for l in eng.layers for p in l.parameters() if getattr(p, '_cpu_pinned', None) is not None
            and p._cpu_pinned.is_pinned())
n_all = sum(1 for l in eng.layers for p in l.parameters())
print(f'[np] 层 _cpu_pinned 状态: {n_pin}/{n_all} pinned', flush=True)

hf0 = hardfault_ps(2)
print(f'[np] 空闲硬缺页: {hf0:,.0f} pages/s', flush=True)

t0 = time.time()
for _ in range(5):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
dt = time.time() - t0
hf1 = hardfault_ps(3)
print(f'[np] decode 5: {dt/5:.2f} s/tok | 硬缺页: {hf1:,.0f} pages/s = {hf1*4096/1e6:.0f} MB/s', flush=True)
print('[np] DONE (对比: pinned版 decode 3.4s, 硬缺页605MB/s)', flush=True)
