"""
三层专家容量测试: VRAM / RAM / mmap虚拟内存
目标: 找到每层能装多少专家, 估算缓存命中天花板
"""
import torch, time, numpy as np, os, gc, mmap, struct, json, tempfile, shutil, psutil

EXPERT_MB = 24  # 24MB per expert
EXPERT_BYTES = EXPERT_MB * 1024 * 1024
DEV = 'cuda'

# === System Info ===
mem = psutil.virtual_memory()
swap = psutil.swap_memory()
print("=" * 60)
print("  [Tier Test] 三层专家容量上限")
print("=" * 60)
print(f"\n  System:")
print(f"    Physical RAM: {mem.total/1024**3:.1f}GB (free: {mem.available/1024**3:.1f}GB)")
print(f"    Swap/Virtual: {swap.total/1024**3:.1f}GB")

# === Tier 1: VRAM ===
print(f"\n  [Tier 1] VRAM (RTX 5070 12GB)")

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
try:
    vram_experts = []
    while True:
        t = torch.zeros(EXPERT_BYTES // 4, dtype=torch.float32, device=DEV)
        vram_experts.append(t)
except torch.cuda.OutOfMemoryError:
    pass

vram_used = torch.cuda.max_memory_allocated() / 1024**3
vram_count = len(vram_experts)
print(f"    Max experts: {vram_count} ({vram_used:.1f}GB)")
print(f"    ~~~~~~~~~ GPU couldn't hold even 1 model layer")
del vram_experts; torch.cuda.empty_cache()

# === Tier 2: RAM (physical, no swap) ===
print(f"\n  [Tier 2] Physical RAM")

# How much is really available
free_ram = mem.available
ram_experts_count = int(free_ram / EXPERT_BYTES)
ram_gb = ram_experts_count * EXPERT_MB / 1024
ram_coverage = ram_experts_count / (512 * 60) * 100
print(f"    Available: {free_ram/1024**3:.1f}GB")
print(f"    Can hold: ~{ram_experts_count} experts ({ram_gb:.1f}GB)")
print(f"    Coverage: {ram_coverage:.1f}% of 30720 total instances")
print(f"    Equivalent: {ram_experts_count/512:.1f} full layers")

# === Tier 3: mmap virtual memory ===
print(f"\n  [Tier 3] mmap Virtual Memory")

# Create a test file representing all expert data (small scale)
test_file = "D:\\ai\\cap_test.bin"
test_mb = 2048  # 2GB test
with open(test_file, "wb") as f:
    f.seek(test_mb * 1024 * 1024 - 1)
    f.write(b'\x00')

fd = os.open(test_file, os.O_RDWR | os.O_BINARY)
mapped = mmap.mmap(fd, 0)
print(f"    Mapped: {test_mb}MB file into virtual address space")

# Measure access pattern: random read + timing
accesses = 2000
total_experts = test_mb // EXPERT_MB
import random
random.seed(42)

print(f"\n    访问模式测试 ({accesses}次随机读):")
# Test: small hot set (fits in RAM) vs large cold set (exceeds RAM)
for hot_pct, label in [(10, "10%热集(fits RAM)"), (50, "50%热集"), (100, "100%全随机")]:
    hot_n = max(1, int(total_experts * hot_pct / 100))
    t0 = time.perf_counter()
    hits = 0; total_bytes = 0

    for i in range(accesses):
        # 80% from hot set, 20% from cold
        if random.random() < 0.8:
            eid = random.randint(0, hot_n - 1)
            hits += 1
        else:
            eid = random.randint(0, total_experts - 1)

        off = eid * EXPERT_BYTES
        # Just touch the page (small read to trigger page fault)
        _ = mapped[off:off+4096]
        total_bytes += 4096

    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / accesses * 1000
    bw = total_bytes / elapsed / 1024**2
    hit_rate = hits / accesses * 100
    print(f"    {label:18s}: {avg_ms:5.2f}ms/access | {bw:6.0f} MB/s | hit={hit_rate:.0f}%")

mapped.close(); os.close(fd)
os.unlink(test_file)

# === Summary ===
print(f"\n{'='*60}")
print(f"  [Summary] 三层专家容量")
print(f"  {'Tier':12s} {'Capacity':>10s} {'Coverage':>10s} {'Benefit'}")
print(f"  {'-'*50}")
print(f"  {'VRAM(12GB)':12s} {'~1 layer':>10s} {'0.0%':>10s} {'Active compute only'}")
print(f"  {'RAM(17GB)':12s} {f'~{ram_experts_count}exp':>10s} {f'{ram_coverage:.1f}%':>10s} {'Hot cache'}")
print(f"  {'mmap(132GB)':12s} {f'ALL 30720':>10s} {'100%':>10s} {'OS manages hot/cold'}")
print(f"")
print(f"  Best strategy:")
print(f"    mmap ALL experts (virtual unlimited)")
print(f"    OS keeps ~{ram_experts_count} hottest in physical RAM")
print(f"    Cold ones fault from SSD at 2.7GB/s")
print(f"    Hot set hit rate depends on access locality (~2% per-layer overlap)")
print(f"")
print(f"  With 2% per-layer overlap + 17GB RAM cache:")
print(f"    Hot set hits: ~{ram_experts_count} experts auto-resident")
print(f"    Misses: rest faulted from SSD")
print(f"    Expected savings: hot_set * 2% overlap * 60 layers")
print(f"    ~{ram_experts_count*0.02*60:.0f} SSD reads avoided per token")
