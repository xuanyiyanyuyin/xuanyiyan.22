# -*- coding: utf-8 -*-
"""双盘读取速度基准: 顺序 / 随机(12MB专家块) / 并发(12线程) — 用 open+seek+read 兼容 Windows"""
import os, sys, time, threading, statistics, random
sys.stdout.reconfigure(encoding='utf-8')

CHUNK = 12 * 1024 * 1024  # 12MB (专家块大小, 与引擎 pread 一致)
N_RAND = 100

def read_at(path, offset, size):
    with open(path, 'rb') as f:
        f.seek(offset)
        return len(f.read(size))

def bench_seq(path, chunk=1*1024*1024):
    """顺序读 1MB 块"""
    sz = os.path.getsize(path)
    t0 = time.time()
    off = 0
    while off < sz:
        read_at(path, off, min(chunk, sz - off))
        off += chunk
    dt = time.time() - t0
    return sz / dt / 2**30

def bench_rand(path, n=N_RAND, chunk=CHUNK):
    """随机 12MB 块 (模拟专家读取)"""
    sz = os.path.getsize(path)
    offsets = [int((sz - chunk) * random.random()) for _ in range(n)]
    lat = []
    t0 = time.time()
    for off in offsets:
        t1 = time.time()
        read_at(path, off, chunk)
        lat.append(time.time() - t1)
    dt = time.time() - t0
    return (n * chunk) / dt / 2**30, statistics.mean(lat) * 1000

def bench_concurrent(path, n_threads=12, n=120, chunk=CHUNK):
    """并发随机读 (模拟 expert_pool 12线程)"""
    sz = os.path.getsize(path)
    offsets = [int((sz - chunk) * random.random()) for _ in range(n)]
    results = [0] * n
    def worker(idx):
        with open(path, 'rb') as f:
            f.seek(offsets[idx])
            results[idx] = len(f.read(chunk))
    t0 = time.time()
    for i in range(0, n, n_threads):
        batch = range(i, min(i + n_threads, n))
        ts = [threading.Thread(target=worker, args=(j,)) for j in batch]
        for t in ts: t.start()
        for t in ts: t.join()
    dt = time.time() - t0
    return sum(results) / dt / 2**30

def run(label, path):
    print(f'\n=== {label} ===', flush=True)
    print(f'  文件: {path} ({os.path.getsize(path)/2**30:.1f}GB)', flush=True)
    seq = bench_seq(path)
    print(f'  顺序读(1MB块): {seq:.2f} GB/s', flush=True)
    rgbs, rms = bench_rand(path)
    print(f'  随机12MB: {rgbs:.2f} GB/s (平均延迟 {rms:.1f}ms)', flush=True)
    cgbs = bench_concurrent(path)
    print(f'  并发12线程随机: {cgbs:.2f} GB/s', flush=True)
    return {'seq': seq, 'rand': rgbs, 'conc': cgbs, 'rand_ms': rms}

if __name__ == '__main__':
    D_FILE = r'D:/ai_v4flash/model-00001-of-00046.safetensors'
    C_FILE = r'C:/ai_diskbench/test.bin'
    os.makedirs(r'C:/ai_diskbench', exist_ok=True)
    if not os.path.exists(C_FILE) or os.path.getsize(C_FILE) < 2*1024**3:
        print('创建 C 盘测试文件 (2GB)...', flush=True)
        with open(C_FILE, 'wb') as f:
            f.seek(2*1024**3 - 1)
            f.write(b'\0')
    r = {}
    r['D盘(致态TiPlus5000)'] = run('D盘 (致态TiPlus5000 1TB, PCIe3.0)', D_FILE)
    r['C盘(SK Hynix)'] = run('C盘 (SK Hynix 512GB, PCIe3.0)', C_FILE)
    print('\n=== 汇总 ===', flush=True)
    for k, v in r.items():
        print(f'  {k}: 顺序 {v["seq"]:.2f} | 随机12MB {v["rand"]:.2f} | 并发12线程 {v["conc"]:.2f} GB/s | 随机延迟 {v["rand_ms"]:.1f}ms', flush=True)
    print('DONE', flush=True)
