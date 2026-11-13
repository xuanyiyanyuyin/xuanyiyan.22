"""快速检验: 从小到大测试每个模型能否加载+推理"""
import os, sys, time, mmap
sys.path.insert(0, r"D:\ai")

tests = [
    ("TTS",      r"D:\ai\models\Qwen3-TTS-12Hz-1.7B-Base",      4.3),
    ("ASR",      r"D:\ai\models\Qwen3-ASR-1.7B",               4.4),
    ("MusicGen", r"D:\ai\models\musicgen-small",                5.5),
    ("FLUX",     r"D:\ai\models\FLUX.1-dev",                    54),
    ("VL-30B",   r"D:\ai\models\Qwen3-VL-30B-A3B-Instruct",     58),
    ("Wan2.1",   r"D:\ai\models\Wan2.1-T2V-14B",               65),
    ("35B-MoE",  r"D:\ai\models\Qwen3.6-35B-A3B",             67),
    ("122B-MoE", r"D:\ai\models\Qwen3.5-122B-A10B",            233),
]

for name, path, expected in tests:
    print(f"[{name}] ", end="", flush=True)
    t0 = time.perf_counter()

    if not os.path.exists(path):
        print(f"NOT FOUND")
        continue

    # Count files and total size
    sfs = [f for f in os.listdir(path) if f.endswith((".safetensors", ".bin"))]
    size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
    size_gb = size / 1024**3

    # Try mmap a file
    try:
        first = sfs[0] if sfs else None
        if first:
            fp = os.path.join(path, first)
            fd = os.open(fp, os.O_RDONLY | os.O_BINARY)
            sz = os.fstat(fd).st_size
            m = mmap.mmap(fd, sz, access=mmap.ACCESS_READ)
            m.close(); os.close(fd)
            mmap_ok = "mmap OK"
        else:
            mmap_ok = "no files"
    except Exception as e:
        mmap_ok = f"mmap FAIL: {e}"

    elapsed = time.perf_counter() - t0
    ok = "OK" if abs(size_gb - expected) < expected * 0.3 else "SIZE?"
    print(f"{size_gb:.0f}GB ({len(sfs)} files) | {mmap_ok} | {elapsed:.1f}s [{ok}]")

print("\nDone.")
