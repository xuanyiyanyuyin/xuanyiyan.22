# -*- coding: utf-8 -*-
"""V4Flash 内存实测: 系统 RAM 余量 + 进程 RSS + VRAM + 缓存配置"""
import sys, time, subprocess, psutil
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch

vm = psutil.virtual_memory()
print(f'[mem] 系统 RAM: 总 {vm.total/2**30:.0f}GB / 已用 {vm.used/2**30:.1f}GB / 可用 {vm.available/2**30:.1f}GB', flush=True)

import v4flash_chunked_sched as vs
print('[mem] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
torch.cuda.synchronize()
dt = time.time() - t0
p = psutil.Process()
rss = p.memory_info().rss / 2**30
print(f'[mem] 引擎加载 {dt:.0f}s', flush=True)
print(f'[mem] 进程 RSS: {rss:.1f}GB', flush=True)

vm2 = psutil.virtual_memory()
print(f'[mem] 加载后系统可用: {vm2.available/2**30:.1f}GB (余量 {vm2.available/2**30:.1f}GB)', flush=True)
print(f'[mem] 进程可再加缓存 ≈ {vm2.available/2**30 - 1:.1f}GB', flush=True)

rr = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
print(f'[mem] VRAM: {rr.stdout.strip()}', flush=True)
print(f'[mem] gpu_exp_max={eng.gpu_exp_max}  expert_lru_max={eng.expert_lru.max}', flush=True)
print('[mem] DONE', flush=True)
