# -*- coding: utf-8 -*-
"""read_expert 微观分解 — 每专家 6 次 pread (w1/s1/w2/s2/w3/s3) 各 tensor 耗时
判断慢因: IO 次数固定成本(小tensor也慢) vs 数据量(大tensor慢小tensor快) vs 总数据
"""
import sys, time, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs

print('[rd] 初始化 ExpertReader...', flush=True)
reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)

# 选 20 个随机专家, 逐 tensor 计时 (单线程, 空闲系统)
N = 20
random.seed(7)
lids = random.choices(range(43), k=N)
eids = random.choices(range(256), k=N)

# 预热
reader.read_expert(lids[0], eids[0])

t_tot = {'w1': [], 's1': [], 'w2': [], 's2': [], 'w3': [], 's3': []}
for i in range(N):
    lid, eid = lids[i], eids[i]
    p = f'layers.{lid}.ffn.experts.{eid}'
    for name, key in [('w1', f'{p}.w1.weight'), ('s1', f'{p}.w1.scale'),
                      ('w2', f'{p}.w2.weight'), ('s2', f'{p}.w2.scale'),
                      ('w3', f'{p}.w3.weight'), ('s3', f'{p}.w3.scale')]:
        t0 = time.time()
        t = reader._read(key)
        t_tot[name].append((time.time() - t0, t.numel() * t.element_size() / 1e6))

print('[rd] 各 tensor 耗时 (20 个专家, 单线程空闲):', flush=True)
for name in ['w1', 's1', 'w2', 's2', 'w3', 's3']:
    dt = [x[0] for x in t_tot[name]]
    mb = [x[1] for x in t_tot[name]]
    n = len(dt)
    print(f'  {name:4s}: avg={sum(dt)/n*1000:6.1f}ms median={sorted(dt)[n//2]*1000:6.1f}ms '
          f'max={max(dt)*1000:6.1f}ms | size={sum(mb)/n:7.1f}MB', flush=True)

# 总 read_expert (6 tensor) 对比
tt = []
for i in range(10):
    t0 = time.time()
    reader.read_expert(lids[i], eids[i])
    tt.append(time.time() - t0)
print(f'[rd] read_expert 总: avg={sum(tt)/len(tt)*1000:.1f}ms (6 tensor/专家)', flush=True)
print('[rd] DONE', flush=True)
