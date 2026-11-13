# -*- coding: utf-8 -*-
"""CPU affinity 精确探针 — 每线程跑在哪个逻辑核? 全部线程覆盖多少核?
区分 GIL (纯Python串行但线程可换核) vs affinity 限制 (线程锁死在少数核)
"""
import sys, ctypes, time, threading, collections
sys.stdout.reconfigure(encoding='utf-8')
def p(*a): print(*a, flush=True)

# 正确读 affinity mask (PDWORD_PTR = 64位)
pm = ctypes.c_ulonglong(0); sm = ctypes.c_ulonglong(0)
ctypes.windll.kernel32.GetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),
                                              ctypes.byref(pm), ctypes.byref(sm))
p(f'进程affinity={pm.value:020b} ({bin(pm.value).count("1")}核) | 系统affinity={sm.value:020b} ({bin(sm.value).count("1")}核)')
p(f'  → 若进程核数 < 系统核数: 进程被 affinity 限制')

# 每线程报告当前所在逻辑核
core_usage = collections.defaultdict(lambda: collections.Counter())
def worker(tid):
    for _ in range(300):
        c = ctypes.windll.kernel32.GetCurrentProcessorNumber()
        core_usage[tid][c] += 1
        time.sleep(0.001)

p('\n=== 8 线程各落在哪些核 (300采样) ===')
ths = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
for t in ths: t.start()
for t in ths: t.join()

all_cores = set()
for tid, cnt in core_usage.items():
    top = cnt.most_common(3)
    core_str = ' '.join(f'核{c}({n})' for c, n in top)
    all_cores.update(cnt.keys())
    p(f'  线程{tid}: 主要 {core_str}  (共{len(cnt)}个核)')
p(f'\n8线程共覆盖 {len(all_cores)} 个逻辑核: {sorted(all_cores)}')

# 单线程跑满采样 (看能否自由迁移)
p('\n=== 单线程可到达的核 (自由迁移测试) ===')
seen = set()
def wander():
    for _ in range(500):
        seen.add(ctypes.windll.kernel32.GetCurrentProcessorNumber())
        time.sleep(0.0005)
w = threading.Thread(target=wander); w.start(); w.join()
p(f'单线程到达 {len(seen)} 核: {sorted(seen)}')
p('=== DONE ===')
