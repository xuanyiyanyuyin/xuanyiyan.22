# -*- coding: utf-8 -*-
"""V4Flash 细粒度 profile — 按算子类型聚合 GPU时间, 拆出 kernel 间隙(调度开销)
跑1个窗口(8层)decode, 对比墙钟 vs GPU算子时间总和
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
from torch.profiler import profile, ProfilerActivity
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

print('[prof2] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

print('[prof2] 预热完成, profile 1 decode token...', flush=True)
torch.cuda.synchronize()
t_wall0 = time.time()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    torch.cuda.synchronize()
wall = time.time() - t_wall0

# 按算子类型聚合 GPU 时间 (只统计 aten:: 高层算子, 聚合 self cuda time)
from collections import defaultdict
type_time = defaultdict(float)
type_cnt = defaultdict(int)
for evt in prof.key_averages():
    name = evt.key
    if not name.startswith('aten::'):
        continue
    # 简化算子名: 取 aten:: 后主要类型
    cat = name.replace('aten::', '').split('_')[0]
    type_time[cat] += evt.self_device_time_total / 1e6 if hasattr(evt, 'self_device_time_total') else 0
    type_cnt[cat] += evt.count

print(f'\n===== 细粒度算子分析 (1 decode token, 墙钟 {wall:.3f}s) =====', flush=True)
print(f'GPU 算子总时间: {sum(type_time.values()):.3f}s', flush=True)
gap = max(0, wall - sum(type_time.values()))
print(f'**Kernel 间隙/调度/同步: {gap:.3f}s ({gap/wall*100:.0f}%)** ← 非GPU计算时间', flush=True)
print(f'  GPU算力实际干活: {sum(type_time.values()):.3f}s ({sum(type_time.values())/wall*100:.0f}%)', flush=True)

print('\n按算子类型 (GPU self time):', flush=True)
for cat, t in sorted(type_time.items(), key=lambda x: -x[1]):
    if t > 0.001:
        print(f'  {cat:<20} {t:8.3f}s ({t/wall*100:5.1f}%)  次数{type_cnt[cat]}', flush=True)

print('\n[prof2] DONE', flush=True)
