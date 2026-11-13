"""
全模型下载脚本: AI动画生产流水线 (7个模型)
国内直连优先: ModelScope > hf-mirror > HuggingFace
"""
import os, subprocess, sys, time

MODEL_DIR = r"D:\ai\models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs(MODEL_DIR, exist_ok=True)

models = [
    # (name, source, model_id, local_dir)
    ("Qwen3.5-122B-A10B", "modelscope", "Qwen/Qwen3.5-122B-A10B", "qwen3.5-122b"),
    ("Qwen3-VL-30B-A3B", "modelscope", "Qwen/Qwen3-VL-30B-A3B", "qwen3-vl-30b"),
    ("Qwen3.6-35B-A3B", "modelscope", "Qwen/Qwen3.6-35B-A3B", "qwen3.6-35b"),
    ("Qwen3-Audio-7B", "modelscope", "Qwen/Qwen3-Audio-7B", "qwen3-audio-7b"),
    ("FLUX.1-dev", "hf-mirror", "black-forest-labs/FLUX.1-dev", "flux1-dev"),
    ("Wan2.1-14B", "hf-mirror", "Wan-AI/Wan2.1-T2V-14B", "wan2.1-14b"),
    ("MusicGen-Large", "hf-mirror", "facebook/musicgen-large", "musicgen-large"),
]

print("=" * 60)
print("  AI动画流水线 · 全模型下载")
print(f"  目标: {MODEL_DIR}")
print(f"  共{len(models)}个模型")
print("=" * 60)

results = []
for name, source, model_id, local_dir in models:
    local_path = os.path.join(MODEL_DIR, local_dir)
    print(f"\n[{name}] {source}: {model_id}")
    t0 = time.perf_counter()

    try:
        if source == "modelscope":
            cmd = [
                sys.executable, "-m", "modelscope", "download",
                "--model", model_id,
                "--local_dir", local_path,
            ]
        else:  # hf-mirror
            cmd = [
                sys.executable, "-c",
                f"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model_id}', local_dir='{local_path}', "
                f"resume_download=True, max_workers=8)"
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            size = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, files in os.walk(local_path) for f in files) / 1024**3
            results.append(f"  [OK] {name}: {size:.1f}GB in {elapsed:.0f}s")
            print(f"  [OK] {size:.1f}GB in {elapsed:.0f}s")
        else:
            results.append(f"  [FAIL] {name}: {result.stderr[:200]}")
            print(f"  [FAIL] {result.stderr[:200]}")
    except Exception as e:
        results.append(f"  [FAIL] {name}: {e}")
        print(f"  [FAIL] {e}")

print("\n" + "=" * 60)
print("  Download Summary:")
for r in results: print(r)
