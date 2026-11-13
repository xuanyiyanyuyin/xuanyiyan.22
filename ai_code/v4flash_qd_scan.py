# -*- coding: utf-8 -*-
"""QD 扫描探针 — 验证"提高专家读取并发度(QD)是否能提速"这一核心假设
模拟一轮 decode 的专家冷读 (157 个随机专家 ≈ 52% miss), 测 QD=1/2/4/8/16 的吞吐
如果 QD 提高吞吐不涨 → 磁盘随机读已饱和, io_uring/QD 方案无效
如果 QD 提高吞吐显著涨 → 有空间, 值得换异步 IO
"""
import sys, time, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
from concurrent.futures import ThreadPoolExecutor
import v4flash_chunked_sched as vs

print('[qd] 初始化 ExpertReader (不加载引擎)...', flush=True)
reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)

# 模拟一轮 decode: 43层 × 活性7专家, 52% miss ≈ 157 个冷读专家
N_EXP = 157
random.seed(42)
lids = random.choices(range(43), k=N_EXP)
eids = random.choices(range(256), k=N_EXP)
EXP_MB = 12  # 每专家 12MB (w1+w2+w3 FP8)

# 预热: 冷启动读一遍 (填 LRU header 缓存, 避免首轮文件句柄开销干扰)
_pool = ThreadPoolExecutor(max_workers=4)
for i in range(8):
    _pool.submit(reader.read_expert, lids[i], eids[i])
_pool.shutdown()

print(f'[qd] 每轮读 {N_EXP} 个随机专家 ≈ {N_EXP*EXP_MB/1e3:.1f}GB', flush=True)
for qd in [1, 2, 4, 8, 16, 32]:
    # 每轮用不同的随机专家, 避免页缓存命中 (真实冷读)
    random.seed(100 + qd)
    _l = random.choices(range(43), k=N_EXP)
    _e = random.choices(range(256), k=N_EXP)
    pool = ThreadPoolExecutor(max_workers=qd)
    t0 = time.time()
    futs = [pool.submit(reader.read_expert, _l[i], _e[i]) for i in range(N_EXP)]
    for f in futs:
        f.result()
    dt = time.time() - t0
    gb = N_EXP * EXP_MB / 1e9
    print(f'  QD={qd:2d}: {dt:6.2f}s | {gb/dt:5.2f} GB/s | {N_EXP/dt:5.0f} op/s | avg={dt/N_EXP*1000:5.1f}ms/op', flush=True)
    pool.shutdown()
print('[qd] DONE', flush=True)
