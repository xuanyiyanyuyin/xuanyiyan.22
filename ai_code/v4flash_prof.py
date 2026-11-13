# -*- coding: utf-8 -*-
"""V4Flash torch.profiler — 定位每层 H2D/D2H 传输来源和大小
跑 2 token, 分析 to/copy_/cpu 等传输操作的时间+字节
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
from torch.profiler import profile, ProfilerActivity
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[prof] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好,介绍一下你自己", return_tensors='pt').input_ids[:, :20].to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

print('[prof] 预热完成, 开始 profile 2 token...', flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(2):
        A = logits[:, -1, :].argmax(-1, keepdim=True)
        logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    torch.cuda.synchronize()

# 传输类操作汇总
print('\n===== 传输类操作 (to/copy/cpu) =====', flush=True)
print(f"{'操作':<35} {'次数':>5} {'CPU总ms':>9} {'CUDA总ms':>9} {'输入MB':>9}", flush=True)
total_h2d = 0.0
total_d2h = 0.0
for evt in prof.key_averages():
    name = evt.key
    if any(k in name for k in ['to', 'copy', 'cpu', 'clone', 'index']):
        cnt = evt.count
        cpu_ms = evt.self_cpu_time_total / 1000
        cuda_ms = evt.self_device_time_total / 1000 if hasattr(evt, 'self_device_time_total') else 0
        inp_mb = evt.input_shapes[0][0].numel() * evt.input_shapes[0][0].element_size() / 2**20 if evt.input_shapes and evt.input_shapes[0] else 0
        print(f"{name:<35} {cnt:>5} {cpu_ms:>9.1f} {cuda_ms:>9.1f} {inp_mb:>9.1f}", flush=True)

print('\n===== 每token传输估算 =====', flush=True)
print('H2D: 43层 × 每层权重(.to(device))', flush=True)
print('D2H: 43层 × 每层权重(.cpu()回传)', flush=True)
# 估算层权重大小
print(f'单层权重估算: 从 layer_ram 43层共13GB → 每层~{13/43:.1f}GB', flush=True)

print('\n[prof] DONE', flush=True)
