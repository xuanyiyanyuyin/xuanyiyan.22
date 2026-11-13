"""
SSD GB级批量读取测试: 找到带宽饱和点
测试从 24MB(1专家) 到 12GB(512专家) 的读取效率
"""
import os, struct, json, time, tempfile, shutil, gc

EXPERT_GU = 16*1024*1024  # 16MB per expert gate_up
EXPERT_DN =  8*1024*1024  #  8MB per expert down
EXPERT_TOT = 24*1024*1024  # 24MB total per expert
NUM_EXPERTS = 512
N_ROUNDS = 5  # 每档测5轮取平均
TARGET_SSD = "D:\\ai\\ssd_bench_dummy.safetensors"  # 测哪个盘就写哪里

print("=" * 70)
print("SSD GB 级批量读取带宽测试")
print(f"目标盘: {TARGET_SSD}")
print(f"测试范围: 1专家(24MB) → 512专家(12GB)")
print("=" * 70)

# === 创建 dummy 文件 ===
print("\n[1/2] 创建 12GB dummy safetensors ...")
t0 = time.perf_counter()

hdr = json.dumps({
    f"model.layers.0.experts": {
        "dtype": "BF16", "shape": [NUM_EXPERTS, 2048+1024, 4096],
        "data_offsets": [0, NUM_EXPERTS * EXPERT_TOT]
    }
}).encode()
hdr_len = len(hdr)
total_data = NUM_EXPERTS * EXPERT_TOT  # 12GB

with open(TARGET_SSD, "wb") as f:
    f.write(struct.pack("<Q", hdr_len))
    f.write(hdr)
    # Write random data to force real allocation (not sparse)
    chunk = 1024*1024  # 1MB chunks
    data = os.urandom(chunk)
    written = 0
    while written < total_data:
        write_size = min(chunk, total_data - written)
        f.write(data[:write_size])
        written += write_size
        if written % (1024*1024*1024) == 0:
            print(f"  {written/1024**3:.0f}GB...", flush=True)

elapsed_write = time.perf_counter() - t0
file_size = os.path.getsize(TARGET_SSD)
print(f"  写入完成: {file_size/1024**3:.1f}GB in {elapsed_write:.0f}s ({file_size/elapsed_write/1024**2:.0f} MB/s)")

# === 测试 ===
print(f"\n[2/2] 批量读取测试 ({N_ROUNDS}轮取平均)")
print("-" * 70)
print(f"{'批量大小':>14s} | {'数据量':>8s} | {'耗时':>8s} | {'带宽':>10s} | {'相对1专家'}")
print("-" * 70)

fd = os.open(TARGET_SSD, os.O_RDONLY | os.O_BINARY)
data_start = hdr_len + 8  # header end

results = []
batch_sizes = [1, 5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 512]

for n_experts in batch_sizes:
    read_size = n_experts * EXPERT_TOT
    # Pre-warm (avoid cold cache penalty)
    os.lseek(fd, data_start, os.SEEK_SET)
    os.read(fd, read_size)

    # Timed reads
    times = []
    for rnd in range(N_ROUNDS):
        os.lseek(fd, data_start, os.SEEK_SET)
        t0 = time.perf_counter()
        buf = os.read(fd, read_size)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    throughput = read_size / avg_time / 1024**2  # MB/s
    size_str = f"{read_size/1024**2:.0f}MB" if read_size < 1024**3 else f"{read_size/1024**3:.1f}GB"

    if n_experts == 1:
        baseline_time = avg_time
        ratio = "1.00x"
    else:
        ratio = f"{baseline_time * n_experts / avg_time:.1f}x"

    print(f"{n_experts:>5}专家({size_str:>5s}) | {read_size/1024**2:6.0f}MB | {avg_time*1000:5.0f}ms | {throughput:7.0f} MB/s | {ratio}")
    results.append((n_experts, read_size, avg_time, throughput))
    gc.collect()

os.close(fd)

# === 总结 ===
print("\n" + "=" * 70)
print("分析")

# Find peak throughput
best = max(results, key=lambda x: x[3])
print(f"  峰值带宽: {best[3]:.0f} MB/s @ {best[0]}专家 ({best[1]/1024**2:.0f}MB)")
print(f"  1专家基线: {results[0][2]*1000:.0f}ms ({results[0][3]:.0f} MB/s)")

# Estimate real impact
peak_ratio = best[3] / results[0][3]
expert_per_layer = 10
current_ms = results[results.index([r for r in results if r[0] >= 10][0])][2] * 1000
print(f"\n  10专家(当前): {current_ms:.0f}ms/层")
print(f"  峰值效率比单读: {peak_ratio:.1f}x")
print(f"\n  结论: GB级批量读{'有' if peak_ratio > 1.3 else '无'}显著收益 (NVMe随机≈顺序)")

shutil.rmtree(TARGET_SSD, ignore_errors=True)
os.unlink(TARGET_SSD) if os.path.exists(TARGET_SSD) else None
print(f"\n临时文件已清理")
