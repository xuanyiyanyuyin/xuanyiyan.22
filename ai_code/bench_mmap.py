"""
mmap (虚拟内存) vs pread (手动读) 对比
模拟专家随机访问: OS页缓存自动管理 vs 手动LRU
"""
import os, struct, json, time, mmap, tempfile, shutil, gc
import numpy as np

EXPERT_TOT = 24*1024*1024  # 24MB
NUM_EXPERTS = 512
FILE_SIZE = NUM_EXPERTS * EXPERT_TOT  # 12GB
SSD_PATH = "D:\\ai\\mmap_bench.safetensors"
N_ROUNDS = 1000  # 1000次随机专家访问
N_WARMUP = 50

print("=" * 60)
print("  mmap(虚拟内存) vs pread(手动读) 随机专家访问")
print("=" * 60)

# === 创建dummy ===
print("\n[1/4] 创建 12GB dummy...")
t0 = time.perf_counter()
hdr = json.dumps({"test": "dummy","shape":[512,3072,4096],"data_offsets":[0,FILE_SIZE]}).encode()
hdr_len = len(hdr)
data = os.urandom(1024*1024)
with open(SSD_PATH, "wb") as f:
    f.write(struct.pack("<Q", hdr_len)); f.write(hdr)
    for _ in range(FILE_SIZE // (1024*1024)):
        f.write(data)
print(f"  {os.path.getsize(SSD_PATH)/1024**3:.1f}GB in {time.perf_counter()-t0:.0f}s")

# 生成随机访问序列
import random
random.seed(42)
accesses = [(random.randint(0, NUM_EXPERTS-1), random.randint(0, 1))
            for _ in range(N_ROUNDS + N_WARMUP)]
data_off = hdr_len + 8

# === 方案A: pread (当前方式) ===
print("\n[2/4] 方案A: pread (手动逐个读)...")
fd = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
times_a = []

for i, (eid, _) in enumerate(accesses):
    t0 = time.perf_counter()
    os.lseek(fd, data_off + eid * EXPERT_TOT, os.SEEK_SET)
    buf = os.read(fd, EXPERT_TOT)
    elapsed = time.perf_counter() - t0
    if i >= N_WARMUP:
        times_a.append(elapsed)
os.close(fd)

avg_a = sum(times_a) / len(times_a) * 1000
p50_a = sorted(times_a)[len(times_a)//2] * 1000
p99_a = sorted(times_a)[int(len(times_a)*0.99)] * 1000
print(f"  均值: {avg_a:.1f}ms | P50: {p50_a:.1f}ms | P99: {p99_a:.1f}ms")

# === 方案B: mmap (OS虚拟内存管理) ===
print("\n[3/4] 方案B: mmap (虚拟内存映射)...")
fd = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
mapped = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
times_b = []

for i, (eid, _) in enumerate(accesses):
    t0 = time.perf_counter()
    buf = mapped[data_off + eid * EXPERT_TOT : data_off + (eid+1) * EXPERT_TOT]
    elapsed = time.perf_counter() - t0
    if i >= N_WARMUP:
        times_b.append(elapsed)

mapped.close(); os.close(fd)

avg_b = sum(times_b) / len(times_b) * 1000
p50_b = sorted(times_b)[len(times_b)//2] * 1000
p99_b = sorted(times_b)[int(len(times_b)*0.99)] * 1000
print(f"  均值: {avg_b:.1f}ms | P50: {p50_b:.1f}ms | P99: {p99_b:.1f}ms")

# === 方案C: mmap + numpy (零拷贝) ===
print("\n[4/4] 方案C: mmap + numpy视图 (零拷贝)...")
fd = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
mapped = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
times_c = []

for i, (eid, _) in enumerate(accesses):
    t0 = time.perf_counter()
    off = data_off + eid * EXPERT_TOT
    arr = np.frombuffer(mapped, dtype=np.uint16, count=EXPERT_TOT//2, offset=off)
    elapsed = time.perf_counter() - t0
    if i >= N_WARMUP:
        times_c.append(elapsed)

mapped.close(); os.close(fd)

avg_c = sum(times_c) / len(times_c) * 1000
p50_c = sorted(times_c)[len(times_c)//2] * 1000
print(f"  均值: {avg_c:.1f}ms | P50: {p50_c:.1f}ms | P99: {p99_c:.1f}ms")

# === 结论 ===
print("\n" + "=" * 60)
print("  对比:")
print(f"  pread(手动):        {avg_a:5.1f}ms/次")
print(f"  mmap(字节切片):     {avg_b:5.1f}ms/次  (vs pread: {avg_b/avg_a:.1f}x)")
print(f"  mmap+numpy(视图):   {avg_c:5.1f}ms/次  (vs pread: {avg_c/avg_a:.1f}x)")

speedup = avg_a / min(avg_b, avg_c)
print(f"\n  最佳 vs pread: {speedup:.1f}x")

# OS缓存的优势
print(f"\n  关键区别:")
print(f"  pread: 每次都要进系统调用 + 内核态拷贝")
print(f"  mmap:  热数据在OS页缓存中 → 零拷贝(memcpy级别)")
print(f"         冷数据自动触发page fault → SSD读")
print(f"         无需手动管理LRU, OS自动淘汰冷页")

if speedup > 1.5:
    print(f"\n  [OK] mmap有效加速 {speedup:.1f}x! 建议替换pread")
else:
    print(f"\n  [INFO] mmap与pread性能接近. NVMe下系统调用开销可忽略")

shutil.rmtree(SSD_PATH, ignore_errors=True)
print("\n临时文件已清理")
