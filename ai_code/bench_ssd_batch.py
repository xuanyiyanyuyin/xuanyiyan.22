"""
40-batch SSD I/O 实测: 对比逐个读 vs 批量读
用 dummy safetensors 模拟真实专家文件布局
"""
import os, struct, json, time, numpy as np, tempfile, shutil

EXPERT_GU = 16*1024*1024  # 16MB
EXPERT_DN =  8*1024*1024  #  8MB
NUM_EXPERTS = 512
N_TEST = 30  # 测试轮数

print("=== 创建 dummy safetensors (模拟 1 层专家布局) ===")
tmpdir = tempfile.mkdtemp(prefix="ssd_bench_")

# 构建 safetensors 格式
tensors = {}
total_gate_up = NUM_EXPERTS * EXPERT_GU  # 8GB
total_down    = NUM_EXPERTS * EXPERT_DN   # 4GB

# gate_up shard
gu_path = os.path.join(tmpdir, "gate_up.safetensors")
gu_hdr = json.dumps({
    f"model.layers.0.experts.gate_up": {
        "dtype": "BF16", "shape": [NUM_EXPERTS, 2048, 4096],
        "data_offsets": [0, total_gate_up]
    }
}).encode()
gu_hdr_len = len(gu_hdr)
with open(gu_path, "wb") as f:
    f.write(struct.pack("<Q", gu_hdr_len))
    f.write(gu_hdr)
    # Write dummy data (zeros, fast)
    f.seek(gu_hdr_len + 8 + total_gate_up - 1)
    f.write(b'\x00')

# down shard
dn_path = os.path.join(tmpdir, "down.safetensors")
dn_hdr = json.dumps({
    f"model.layers.0.experts.down": {
        "dtype": "BF16", "shape": [NUM_EXPERTS, 4096, 1024],
        "data_offsets": [0, total_down]
    }
}).encode()
dn_hdr_len = len(dn_hdr)
with open(dn_path, "wb") as f:
    f.write(struct.pack("<Q", dn_hdr_len))
    f.write(dn_hdr)
    f.seek(dn_hdr_len + 8 + total_down - 1)
    f.write(b'\x00')

gu_size = os.path.getsize(gu_path)
dn_size = os.path.getsize(dn_path)
print(f"  gate_up.safetensors: {gu_size/1024**3:.1f}GB")
print(f"  down.safetensors:    {dn_size/1024**3:.1f}GB")
print(f"  Total: {(gu_size+dn_size)/1024**3:.1f}GB\n")

# === Test 1: 逐个读 10 专家 (当前方式) ===
print("=== 测试 1: 逐个读 10 专家 (当前方式) ===")
fd_gu = os.open(gu_path, os.O_RDONLY | os.O_BINARY)
fd_dn = os.open(dn_path, os.O_RDONLY | os.O_BINARY)
hdr_gu = 8 + gu_hdr_len
hdr_dn = 8 + dn_hdr_len

t0 = time.perf_counter()
for _ in range(N_TEST):
    for eid in range(10):
        os.lseek(fd_gu, hdr_gu + eid * EXPERT_GU, os.SEEK_SET)
        buf = os.read(fd_gu, EXPERT_GU)
        os.lseek(fd_dn, hdr_dn + eid * EXPERT_DN, os.SEEK_SET)
        buf = os.read(fd_dn, EXPERT_DN)
elapsed = time.perf_counter() - t0
per_read = elapsed / (N_TEST * 10)
print(f"  总耗时: {elapsed:.1f}s ({N_TEST}轮×10专家)")
print(f"  每次(含2次read): {per_read*1000:.1f}ms")
print(f"  每层10专家: {per_read*10*1000:.0f}ms")
print(f"  有效带宽: {10*24/(per_read*10)*1024:.0f} MB/s\n")

# === Test 2: 批量读 40 专家 (新方式) ===
print("=== 测试 2: 批量读 40 专家 (新方式) ===")
os.lseek(fd_gu, 0, os.SEEK_SET)  # 重置
os.lseek(fd_dn, 0, os.SEEK_SET)
t0 = time.perf_counter()
for _ in range(N_TEST):
    os.lseek(fd_gu, hdr_gu, os.SEEK_SET)
    buf_gu = os.read(fd_gu, 40 * EXPERT_GU)
    os.lseek(fd_dn, hdr_dn, os.SEEK_SET)
    buf_dn = os.read(fd_dn, 40 * EXPERT_DN)
elapsed = time.perf_counter() - t0
per_batch = elapsed / N_TEST
print(f"  总耗时: {elapsed:.1f}s ({N_TEST}轮×40专家)")
print(f"  每次(2次大块read): {per_batch*1000:.1f}ms")
print(f"  有效带宽: {40*24/elapsed*N_TEST*1024:.0f} MB/s\n")

# === Test 3: 逐个读 40 专家 (对比) ===
print("=== 测试 3: 逐个读 40 专家 (对比) ===")
t0 = time.perf_counter()
for _ in range(N_TEST):
    for eid in range(40):
        os.lseek(fd_gu, hdr_gu + eid * EXPERT_GU, os.SEEK_SET)
        buf = os.read(fd_gu, EXPERT_GU)
        os.lseek(fd_dn, hdr_dn + eid * EXPERT_DN, os.SEEK_SET)
        buf = os.read(fd_dn, EXPERT_DN)
elapsed_ind = time.perf_counter() - t0
print(f"  总耗时: {elapsed_ind:.1f}s ({N_TEST}轮×40专家)")
print(f"  有效带宽: {40*24/elapsed_ind*N_TEST*1024:.0f} MB/s\n")

os.close(fd_gu); os.close(fd_dn)

# === 总结 ===
speedup = elapsed_ind / elapsed
print("=" * 60)
print(f"  40专家 逐个读: {elapsed_ind/N_TEST*1000:.0f}ms/批")
print(f"  40专家 批量读: {elapsed/N_TEST*1000:.0f}ms/批")
print(f"  加速比:       {speedup:.1f}x")
print(f"  每层节省:     {(elapsed_ind-elapsed)/N_TEST*1000:.0f}ms")
print(f"  60层节省:     {(elapsed_ind-elapsed)/N_TEST*60:.1f}s")

# 推算真实推理 (叠加到当前10.9s)
saved = (elapsed_ind - elapsed) / N_TEST * 60
new_time = 10.9 - saved
print(f"\n  当前推理:     10.9s/token")
print(f"  +40-batch后:  {new_time:.1f}s/token")

shutil.rmtree(tmpdir)
print(f"\n临时文件已清理")
