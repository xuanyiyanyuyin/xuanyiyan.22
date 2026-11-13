# -*- coding: utf-8 -*-
"""V4Flash 显存余量检查: 启动后测 VRAM 空闲 + 专家大小 → 可加热专家数"""
import sys, os, json, time, subprocess
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs

def nvsmi():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return tuple(map(int, r.stdout.strip().split(',')))

print('[vram] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
torch.cuda.synchronize()
print(f'[vram] 引擎加载 {time.time()-t0:.0f}s', flush=True)
time.sleep(1)

total, used, free = nvsmi()
alloc = torch.cuda.memory_allocated() / 2**30
reserved = torch.cuda.memory_reserved() / 2**30
print(f'[vram] 显存总 {total}MB / 已用 {used}MB / 空闲 {free}MB', flush=True)
print(f'[vram] torch alloc {alloc:.2f}GB / reserved {reserved:.2f}GB', flush=True)

# 专家大小 (读实际 tensor)
print(f'[vram] gpu_exp_max={eng.gpu_exp_max}  expert_lru={eng.expert_lru.max_items}', flush=True)
from v4flash_chunked_sched import pread
eid = 0
key = f'model.layers.0.mlp.experts.{eid}'
w1 = pread(f'{key}.w1.weight'); w2 = pread(f'{key}.w2.weight'); w3 = pread(f'{key}.w3.weight')
nbytes = (w1.numel()+w2.numel()+w3.numel()) * w1.element_size()
print(f'[vram] 专家单例: w1{list(w1.shape)} w2{list(w2.shape)} w3{list(w3.shape)} = {nbytes/2**20:.1f}MB (FP8)', flush=True)

# 可加热专家数
free_gb = free / 1024
# 保守留 1GB 余量
avail_for_experts = free_gb - 1.0
n_experts = int(avail_for_experts * 1024 / (nbytes/2**20)) if nbytes else 0
print(f'[vram] 空闲 {free_gb:.1f}GB, 留1GB余量后约可再加热 {n_experts} 个专家', flush=True)
print(f'[vram] 当前 gpu_exp_max=50 → 提到 {50+n_experts} 时 GPU 专家显存 ≈ {(50+n_experts)*nbytes/2**30:.1f}GB', flush=True)
print('[vram] DONE', flush=True)
