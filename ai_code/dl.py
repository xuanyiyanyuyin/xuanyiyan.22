"""全模型下载 v2 - nohup持久化 + 断点续传"""
import os, time
from modelscope.hub.snapshot_download import snapshot_download

MODEL_DIR = r"D:\ai\models"
models = [
    ("Qwen3.6-35B-A3B",      "Qwen/Qwen3.6-35B-A3B",             70),
    ("Qwen3-VL-30B-A3B",     "Qwen/Qwen3-VL-30B-A3B-Instruct",   60),
    ("Qwen3-TTS-1.7B",       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",     3),
    ("Qwen3-ASR-1.7B",       "Qwen/Qwen3-ASR-1.7B",               3),
    ("FLUX.1-dev",           "black-forest-labs/FLUX.1-dev",      58),
    ("Wan2.1-T2V-14B",       "Wan-AI/Wan2.1-T2V-14B",            28),
    ("Qwen3.5-122B-A10B",    "Qwen/Qwen3.5-122B-A10B",           244),
]
# ACE-Step v1.5: needs VPN, skip for now

for name, mid, est in models:
    path = f"{MODEL_DIR}/{mid.split('/')[-1]}"
    t0 = time.perf_counter()
    print(f"[{time.strftime('%H:%M:%S')}] {name} (~{est}GB)")
    try:
        snapshot_download(mid, cache_dir=MODEL_DIR, local_dir=path)
        elapsed = time.perf_counter() - t0
        print(f"  [OK] {elapsed:.0f}s")
    except Exception as e:
        print(f"  [FAIL] {e}")
print("Done.")
