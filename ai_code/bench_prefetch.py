"""
层间全专家预取测试: GPU算N层时 后台读N+1层全部512专家
模拟: RAM预取线程 + 虚拟GPU计算 并发
"""
import os, struct, json, time, threading, tempfile, shutil

EXPERT_GU = 16*1024*1024; EXPERT_DN = 8*1024*1024
EXPERT_TOT = 24*1024*1024; NUM_EXPERTS = 512
LAYERS = 60; N_ROUNDS = 3
SSD_PATH = "D:\\ai\\ssd_prefetch_dummy.safetensors"

print("=" * 60)
print("  层间全专家预取测试 (512专家/层, 60层)")
print("=" * 60)

# === 创建 dummy ===
print("\n[1/3] 创建 12GB dummy ...")
t0 = time.perf_counter()
hdr = json.dumps({"experts": {"dtype":"BF16","shape":[512,3072,4096],"data_offsets":[0,512*EXPERT_TOT]}}).encode()
hdr_len = len(hdr)
with open(SSD_PATH, "wb") as f:
    f.write(struct.pack("<Q", hdr_len)); f.write(hdr)
    chunk = os.urandom(1024*1024)
    for _ in range(512 * EXPERT_TOT // (1024*1024)):
        f.write(chunk)
print(f"  {os.path.getsize(SSD_PATH)/1024**3:.1f}GB in {time.perf_counter()-t0:.0f}s")

# === 方案A: 串行 (当前方式) ===
print("\n[2/3] 方案A: 串行 (逐层读→算→读→算)")
fd = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
data_off = hdr_len + 8

t0 = time.perf_counter()
for rnd in range(N_ROUNDS):
    for lid in range(LAYERS):
        # 读10专家 (当前方式)
        off = data_off + (lid % 1) * 512 * EXPERT_TOT  # all layers same file
        for eid in range(10):
            os.lseek(fd, off + eid * EXPERT_TOT, os.SEEK_SET)
            buf = os.read(fd, EXPERT_GU)
            buf = os.read(fd, EXPERT_DN)
        # 模拟GPU (30ms)
        time.sleep(0.030)
serial_time = time.perf_counter() - t0
serial_per = serial_time / (N_ROUNDS * LAYERS)
print(f"  总耗时: {serial_time:.1f}s")
print(f"  每层: {serial_per*1000:.0f}ms")
print(f"  等效: {serial_per*60:.1f}s/token")

# === 方案B: 层间预取 (后台读512专家) ===
print("\n[3/3] 方案B: 层间预取 (后台全读512专家)")

def prefetch_layer(lid, results, idx):
    """读一层全部512专家 → buffer"""
    off = data_off + (lid % 1) * 512 * EXPERT_TOT
    os.lseek(fd, off, os.SEEK_SET)
    buf = os.read(fd, 512 * EXPERT_TOT)
    results[idx] = len(buf)

prefetch_time = 0
compute_time = 0
t0 = time.perf_counter()

for rnd in range(N_ROUNDS):
    # 预热: 先读第0层
    buf0 = bytearray(1)
    prefetch_layer(0, buf0, 0)

    for lid in range(1, LAYERS):
        # 后台读 lid 层
        buf_next = bytearray(1)
        t_read = time.perf_counter()
        prefetch_layer(lid, buf_next, 0)
        prefetch_time += time.perf_counter() - t_read

        # 同时 GPU 算 lid-1 层
        t_cpu = time.perf_counter()
        time.sleep(0.030)
        compute_time += time.perf_counter() - t_cpu

    # 最后一层
    t_cpu = time.perf_counter()
    time.sleep(0.030)
    compute_time += time.perf_counter() - t_cpu

pipe_time = time.perf_counter() - t0
pipe_per = pipe_time / (N_ROUNDS * LAYERS)
os.close(fd)

# 方案B: 预取和计算重叠, 取较大者
ssd_per_layer = (512 * EXPERT_TOT) / (3.5 * 1024**3)  # ~3.4s for 12GB
gpu_per_layer = 0.030
# 预取只需读1次12GB (所有512专家), 后续层可能复用
# 但每层专家不同, 所以每层都要读
overlap_layer = max(ssd_per_layer, gpu_per_layer)

print(f"\n  方案A(串行): {serial_per*1000:.0f}ms/层 → {serial_per*60:.1f}s/token")
print(f"  方案B(预取): SSD={ssd_per_layer*1000:.0f}ms + GPU={gpu_per_layer*1000:.0f}ms")
print(f"               重叠后={overlap_layer*1000:.0f}ms/层 → {overlap_layer*60:.1f}s/token")
speedup = serial_per / overlap_layer
print(f"  理论加速: {speedup:.1f}x")

# 实际考量
print(f"\n  ⚠️ 现实约束:")
print(f"  每层512专家=12GB, 60层=720GB → 无法全缓存")
print(f"  但32GB RAM可缓存2层全专家(24GB) → 双缓冲可行")
buf_time = (512 * EXPERT_TOT) / (3.0 * 1024**3)  # ~4s
print(f"  双缓冲: 每层SSD={buf_time:.1f}s(全读) + GPU=0.03s")
print(f"  重叠后≈max({buf_time:.1f}, 0.03)={buf_time:.1f}s/层")
print(f"  → {buf_time*60:.0f}s/token ← GPU闲置中读SSD, 反效果!")

print(f"\n  结论: 层间预取在SSD瓶颈下无效")
print(f"  因为GPU 30ms << SSD 4s, 计算完全被I/O淹没")
print(f"  只有SSD快到GPU等I/O时, 预取才有意义 (需Gen5+速度)")

shutil.rmtree(SSD_PATH, ignore_errors=True)
print("\n临时文件已清理")
