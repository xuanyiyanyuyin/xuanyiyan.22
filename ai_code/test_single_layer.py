"""
单层管线验证：SSD→RAM→VRAM→CUDA forward 全链路。
"""
import sys; sys.path.insert(0, r'D:\ai')
import time, numpy as np, torch

from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from gpu_engine_minimal import MinimalGPUEngine
from layer_utils import load_layer_types

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'

print("=" * 60)
print("单层管线验证")
print("=" * 60)

# 1) Init
t0 = time.perf_counter()
reader = SafeExpertReader(MODEL)
streamer = NonExpertStreamer(MODEL)
gpu = MinimalGPUEngine()
print(f"[Init] {time.perf_counter()-t0:.1f}s")

# 2) Test full_attention layer only (layer 3 is cleanest)
for layer_id in [3]:
    print(f"\n{'='*40}")
    print(f"Layer {layer_id}")
    print(f"{'='*40}")

    # A) Load non-expert weights from SSD
    t1 = time.perf_counter()
    ne_weights = streamer.read_layer(layer_id)
    ne_mb = sum(v.nbytes for v in ne_weights.values()) / 1e6
    print(f"  [SSD→RAM] {len(ne_weights)} tensors, {ne_mb:.0f}MB in {time.perf_counter()-t1:.2f}s")

    # Show key tensor shapes
    for name, arr in sorted(ne_weights.items())[:5]:
        print(f"    {name.split('.')[-2]}.{name.split('.')[-1]}: {arr.shape}")

    # B) Load 2 experts
    t2 = time.perf_counter()
    gu, dn = reader.read_expert(layer_id, 0)
    gu2, dn2 = reader.read_expert(layer_id, 1)
    gu_stack = np.stack([gu, gu2])
    dn_stack = np.stack([dn, dn2])
    print(f"  [SSD→RAM] 2 experts in {time.perf_counter()-t2:.2f}s, shapes: gu={gu_stack.shape} dn={dn_stack.shape}")

    # C) Copy to VRAM
    t3 = time.perf_counter()
    for name, arr in ne_weights.items():
        gpu.async_copy_h2d(arr, name)
    gpu.sync_stream()
    print(f"  [RAM→VRAM] {time.perf_counter()-t3:.2f}s")

    # D) Forward
    t4 = time.perf_counter()
    hidden = np.random.randn(1, 8, 4096).astype(np.float32) * 0.02  # small init
    result = gpu.forward_layer_test(hidden, ne_weights, gu_stack, dn_stack, layer_id)
    print(f"  [GPU Compute] {time.perf_counter()-t4:.2f}s")

    # E) Cleanup VRAM
    gpu.cleanup()

print(f"\n{'='*60}")
print("ALL OK - SSD→RAM→VRAM→CUDA 全链路验证通过")
print(f"{'='*60}")
reader.close()
streamer.close()
