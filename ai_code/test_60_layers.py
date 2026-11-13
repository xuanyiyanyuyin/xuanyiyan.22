"""
Phase 1: 60 层全栈 forward 验证。
无 KV cache, batch=1, 逐层 SSD→RAM→VRAM→CUDA→验证。
"""
import sys; sys.path.insert(0, r'D:\ai')
import time, numpy as np, torch
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from gpu_engine_minimal import MinimalGPUEngine
from layer_utils import load_layer_types

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'

print("=" * 60)
print("Phase 1: 60 层全栈 forward")
print("=" * 60)

t0 = time.perf_counter()
reader = SafeExpertReader(MODEL)
streamer = NonExpertStreamer(MODEL)
gpu = MinimalGPUEngine()
layer_types = load_layer_types(MODEL)
print(f"[Init] {time.perf_counter()-t0:.1f}s, {len(layer_types)} layers")

# ── OS Cache Warmup: 顺序扫一遍所有 shard，填满 page cache ──
print("[Warmup] Priming OS page cache...")
tw = time.perf_counter()
for lid in range(60):
    plan = streamer._plans[lid]
    for sr in plan.shard_reads:
        fd = streamer._get_fd(sr.shard_path)
        import os as _os
        _os.lseek(fd, sr.abs_read_start, _os.SEEK_SET)
        _os.read(fd, min(65536, sr.read_size))
print(f"[Warmup] Done in {time.perf_counter()-tw:.1f}s")

# ── Test input ──
hidden = np.random.randn(1, 8, 4096).astype(np.float32) * 0.02

# ── Track stats ──
times_ssd = []
times_expert = []
times_h2d = []
times_gpu = []
nan_layers = []
inf_layers = []

print(f"\n{'Layer':>5} {'Type':>17} {'SSD(s)':>8} {'Exp(s)':>8} {'H2D(s)':>8} {'GPU(s)':>8} {'mean':>10} {'std':>10} {'status'}")
print("-" * 95)

for lid in range(60):
    # A) SSD → RAM: non-expert weights
    t1 = time.perf_counter()
    ne_weights = streamer.read_layer(lid)
    t2 = time.perf_counter()
    times_ssd.append(t2 - t1)

    # B) SSD → RAM: experts (simulate: read 2 for testing)
    t3 = time.perf_counter()
    gu, dn = reader.read_expert(lid, 0)
    gu2, dn2 = reader.read_expert(lid, 1)
    gu_stack = np.stack([gu, gu2])
    dn_stack = np.stack([dn, dn2])
    t4 = time.perf_counter()
    times_expert.append(t4 - t3)

    # C) RAM → VRAM
    t5 = time.perf_counter()
    for name, arr in ne_weights.items():
        gpu.async_copy_h2d(arr, name)
    gpu.sync_stream()
    t6 = time.perf_counter()
    times_h2d.append(t6 - t5)

    # D) GPU Compute
    t7 = time.perf_counter()
    result = gpu.forward_layer_test(hidden, ne_weights, gu_stack, dn_stack, lid)
    t8 = time.perf_counter()
    times_gpu.append(t8 - t7)

    # Validate
    has_nan = np.any(np.isnan(result))
    has_inf = np.any(np.isinf(result))
    if has_nan: nan_layers.append(lid)
    if has_inf: inf_layers.append(lid)

    attn_type = layer_types.get(lid, "?")
    status = "NaN" if has_nan else ("Inf" if has_inf else "OK")
    print(f"{lid:5d} {attn_type:>17} {t2-t1:8.3f} {t4-t3:8.3f} {t6-t5:8.3f} {t8-t7:8.3f} "
          f"{result.mean():10.6f} {result.std():10.6f} {status}")

    hidden = result
    gpu.cleanup()

# ── Summary ──
total = sum(times_ssd) + sum(times_expert) + sum(times_h2d) + sum(times_gpu)
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  SSD→RAM:      {sum(times_ssd):.1f}s ({sum(times_ssd)/total*100:.0f}%)")
print(f"  Expert read:  {sum(times_expert):.1f}s ({sum(times_expert)/total*100:.0f}%)")
print(f"  RAM→VRAM:     {sum(times_h2d):.1f}s ({sum(times_h2d)/total*100:.0f}%)")
print(f"  GPU Compute:  {sum(times_gpu):.1f}s ({sum(times_gpu)/total*100:.0f}%)")
print(f"  TOTAL:        {total:.1f}s")
print(f"  NaN layers:   {nan_layers if nan_layers else 'NONE'}")
print(f"  Inf layers:   {inf_layers if inf_layers else 'NONE'}")
print(f"  Final output: mean={hidden.mean():.6f} std={hidden.std():.6f}")

if nan_layers or inf_layers:
    print(f"\n  FAILED: NaN={nan_layers} Inf={inf_layers}")
else:
    print(f"\n  PASSED: All 60 layers clean")

reader.close()
streamer.close()
gpu.cleanup()
torch.cuda.empty_cache()
