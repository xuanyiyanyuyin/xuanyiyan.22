# -*- coding: utf-8 -*-
"""细粒度调度裁决 — 全局 QD 限流 + tensor 级调度 vs 当前 expert_pool6+专家内串行
当前: 专家间并发6, 每专家6 tensor 串行 → 磁盘 QD≈6, tensor间有clone间隙(磁盘空闲)
新方案: 全局信号量锁磁盘 QD=N, 所有 tensor 读经信号量 → 磁盘持续忙碌无间隙
对比各 QD 限流值, 找最优调度
"""
import sys, time, random, threading
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs

reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)

# 模拟引擎: 每 token 磁盘读 234 专家, 每专家 6 tensor
N_EXP = 234
random.seed(42)
lids = random.choices(range(43), k=N_EXP)
eids = random.choices(range(256), k=N_EXP)

def expert_keys(lid, eid):
    p = f'layers.{lid}.ffn.experts.{eid}'
    return [f'{p}.w1.weight', f'{p}.w1.scale', f'{p}.w2.weight', f'{p}.w2.scale',
            f'{p}.w3.weight', f'{p}.w3.scale']

# ① 当前方式: 专家间 QD6, 专家内串行
def sched_current(qd=6):
    pool = ThreadPoolExecutor(max_workers=qd)
    futs = [pool.submit(reader.read_expert, lids[i], eids[i]) for i in range(N_EXP)]
    for f in futs: f.result()
    pool.shutdown()

# ② 全局 QD 限流 + tensor 级并行: 所有 tensor 读经信号量
def sched_qd_limited(qd_limit):
    sem = threading.Semaphore(qd_limit)
    # 提交所有 tensor 读任务, 信号量锁磁盘并发
    pool = ThreadPoolExecutor(max_workers=qd_limit * 3)  # 足够线程, 信号量才是瓶颈
    def limited_read(key):
        with sem:
            return reader._read(key)
    futs = []
    for i in range(N_EXP):
        for k in expert_keys(lids[i], eids[i]):
            futs.append(pool.submit(limited_read, k))
    for f in futs: f.result()
    pool.shutdown()

# ③ 全局 QD 限流 + 专家内串行 (tensor 级信号量但顺序提交)
def sched_qd_expert(qd_limit):
    sem = threading.Semaphore(qd_limit)
    pool = ThreadPoolExecutor(max_workers=qd_limit * 3)
    def read_expert_limited(lid, eid):
        return tuple(reader._read(k) for k in expert_keys(lid, eid))
    futs = []
    for i in range(N_EXP):
        futs.append(pool.submit(read_expert_limited, lids[i], eids[i]))
    for f in futs: f.result()
    pool.shutdown()

print(f'[ql] 模拟 {N_EXP} 专家磁盘读 ({N_EXP*12.6/1000:.1f}GB), 测 4 种调度:', flush=True)

# 预热 (填 header 缓存)
sched_current(2)

t0 = time.time(); sched_current(6); t1 = time.time()
print(f'[ql] A 当前(专家间QD6+串行):     {t1-t0:.2f}s ({N_EXP*12.6/1000/(t1-t0):.2f}GB/s)', flush=True)

for qd in (4, 6, 8, 12):
    t0 = time.time(); sched_qd_limited(qd); t1 = time.time()
    print(f'[ql] B tensor级限流QD={qd:2d}:       {t1-t0:.2f}s ({N_EXP*12.6/1000/(t1-t0):.2f}GB/s)', flush=True)

for qd in (6, 8):
    t0 = time.time(); sched_qd_expert(qd); t1 = time.time()
    print(f'[ql] C 专家级限流QD={qd:2d}(串行):  {t1-t0:.2f}s ({N_EXP*12.6/1000/(t1-t0):.2f}GB/s)', flush=True)
print('[ql] DONE', flush=True)
