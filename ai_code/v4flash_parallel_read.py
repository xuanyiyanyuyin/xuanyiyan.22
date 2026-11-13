# -*- coding: utf-8 -*-
"""裁决实验 — read_expert 48ms 的瓶颈是 Python 串行还是磁盘?
串行: 6 次 f.read 依次 (当前实现)
并行: 6 线程同时读 6 个 tensor (磁盘可同时处理, Python 不等)
如果并行显著更快 → Python 串行是瓶颈 → 优化方向明确
如果并行无差异 → 磁盘随机读是瓶颈 → 附件假设错误
"""
import sys, time, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import torch
import v4flash_chunked_sched as vs

reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)

def read_serial(lid, eid):
    return reader.read_expert(lid, eid)

def read_parallel(lid, eid):
    p = f'layers.{lid}.ffn.experts.{eid}'
    keys = [f'{p}.w1.weight', f'{p}.w1.scale', f'{p}.w2.weight', f'{p}.w2.scale',
            f'{p}.w3.weight', f'{p}.w3.scale']
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(reader._read, k) for k in keys]
        return tuple(f.result() for f in futs)

# 正确性: 并行结果必须等于串行
print('[pr] 正确性验证 (5 专家):', flush=True)
random.seed(11)
ok = True
for _ in range(5):
    lid, eid = random.randint(0, 42), random.randint(0, 255)
    s = read_serial(lid, eid)
    p = read_parallel(lid, eid)
    for j in range(6):
        d = (p[j].float() - s[j].float()).abs().max().item()
        if d != 0:
            ok = False
            print(f'  ({lid},{eid}): MISMATCH diff={d}', flush=True)
print(f'[pr] {"全部一致!" if ok else "不一致!"}', flush=True)

# 计时: 30 专家, 串行 vs 并行 (专家间也并行, 模拟引擎 QD)
N = 30
lids = random.choices(range(43), k=N)
eids = random.choices(range(256), k=N)

# 模拟引擎: 专家间 QD4 并发, 专家内 串行 vs 并行
def run_batch(fn, workers):
    pool = ThreadPoolExecutor(max_workers=workers)
    t0 = time.time()
    futs = [pool.submit(fn, lids[i], eids[i]) for i in range(N)]
    for f in futs: f.result()
    return time.time() - t0

t_ser = run_batch(read_serial, 4)
t_par = run_batch(read_parallel, 4)
print(f'[pr] 专家间QD4+专家内串行: {t_ser:.2f}s ({t_ser/N*1000:.1f}ms/专家)', flush=True)
print(f'[pr] 专家间QD4+专家内并行: {t_par:.2f}s ({t_par/N*1000:.1f}ms/专家)', flush=True)
print(f'[pr] 并行提速: {(t_ser-t_par)/t_ser*100:.0f}%', flush=True)

# 纯串行 (QD1) vs 并行, 单专家视角
t0 = time.time()
for i in range(N): read_serial(lids[i], eids[i])
t_qd1 = time.time() - t0
t0 = time.time()
for i in range(N): read_parallel(lids[i], eids[i])
t_qd1p = time.time() - t0
print(f'[pr] QD1串行: {t_qd1/N*1000:.1f}ms/专家 | QD1并行(6tensor): {t_qd1p/N*1000:.1f}ms/专家 | 提速 {(t_qd1-t_qd1p)/t_qd1*100:.0f}%', flush=True)
print('[pr] DONE', flush=True)
