# -*- coding: utf-8 -*-
"""安全拷贝 22 个 shard 到 C 盘 (最多 3 并发, 避免 I/O 饱和重启)"""
import os, sys, shutil, threading, time
sys.stdout.reconfigure(encoding='utf-8')

SRC = r'D:/ai_v4flash'
DST = r'C:/ai_v4flash_experts'
os.makedirs(DST, exist_ok=True)

shards = [f'model-{i:05d}-of-00046.safetensors' for i in range(2, 24)]
done = [0]
lock = threading.Lock()

def copy_one(s):
    src = os.path.join(SRC, s)
    dst = os.path.join(DST, s)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        with lock:
            done[0] += 1
            print(f'[{done[0]}/{len(shards)}] {s} 已存在', flush=True)
        return
    t0 = time.time()
    shutil.copyfile(src, dst)
    with lock:
        done[0] += 1
        print(f'[{done[0]}/{len(shards)}] {s} {os.path.getsize(src)/2**30:.1f}GB in {time.time()-t0:.0f}s', flush=True)

# 3 并发
pool = []
for s in shards:
    while len([t for t in pool if t.is_alive()]) >= 3:
        time.sleep(0.5)
    t = threading.Thread(target=copy_one, args=(s,))
    t.start()
    pool.append(t)
for t in pool:
    t.join()
print('=== 拷贝完成 ===', flush=True)
