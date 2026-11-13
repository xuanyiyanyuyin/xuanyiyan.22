# -*- coding: utf-8 -*-
"""CPU 线程检查 v2 — affinity + 线程数扫描 (精简, 全部 flush)
"""
import sys, ctypes, time, threading
sys.stdout.reconfigure(encoding='utf-8')
def p(*a): print(*a, flush=True)

p('=== CPU affinity ===')
hproc = ctypes.windll.kernel32.GetCurrentProcess()
pmask = ctypes.c_size_t(0); smask = ctypes.c_size_t(0)
ctypes.windll.kernel32.GetProcessAffinityMask(hproc, ctypes.byref(pmask), ctypes.byref(smask))
pbits = bin(pmask.value).count('1'); sbits = bin(smask.value).count('1')
p(f'系统可用核: {sbits} | 进程实际可用: {pbits}')
p(f' 进程mask={pmask.value:020b}')
p(f' 系统mask={smask.value:020b}')
if pbits < sbits:
    p(f' ⚠️ 进程被 affinity 限制! 仅 {pbits}/{sbits} 核可用')
else:
    p(f' ✓ 进程无 affinity 限制')

# 环境变量里的 CPU 配置 (CherryStudio/宿主可能注入)
import os
for var in ('NUMBER_OF_PROCESSORS', 'PROCESSOR_IDENTIFIER', 'CLAUDE_CODE_MAX_THREADS', 'UV_THREADPOOL_SIZE'):
    if var in os.environ:
        p(f' {var}={os.environ[var]}')

p('=== 线程数扫描 (CPU密集, 观察并行效果) ===')
def busy():
    t0 = time.time()
    x = 0
    while time.time() - t0 < 0.2:
        x += 1
    return x

for n in (2, 4, 6, 8, 10, 12, 14, 16, 20):
    t0 = time.time()
    ths = [threading.Thread(target=busy) for _ in range(n)]
    for t in ths: t.start()
    for t in ths: t.join()
    dt = time.time() - t0
    # n 个 200ms 任务, 若并行在 P 个核: dt≈0.2*n/eff_cores (GIL 下≈0.2*n)
    eff = n * 0.2 / dt
    p(f' {n:2d}线程: {dt*1000:5.0f}ms | 等效并行核={eff:.1f}')

p('=== DONE ===')
