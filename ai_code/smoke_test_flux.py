"""
[Smoke Test] FLUX.1-dev 冒烟测试
验证: 模型加载 → Text Encoder → DiT去噪 → VAE解码 → 输出
"""
import torch, time, os, hashlib, json

FLUX_DIR = r"D:\ai\models\FLUX.1-dev"
DEV = "cuda"
PROMPT = "A red apple on wooden table, studio lighting, 8k"
STEPS = 20  # 少步数快速验证
OUTPUT = r"D:\ai\output\smoke_test_flux.png"

os.makedirs(r"D:\ai\output", exist_ok=True)

print("=" * 60)
print("  [Smoke Test] FLUX.1-dev")
print(f"  Prompt: {PROMPT}")
print(f"  Steps: {STEPS}")
print("=" * 60)

# ====== 1. Check files ======
print("\n[1/5] Check model files...")
required = ["flux1-dev.safetensors", "ae.safetensors"]
for d in ["text_encoder_2", "vae", "transformer"]:
    path = os.path.join(FLUX_DIR, d)
    if os.path.isdir(path):
        sfs = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        print(f"  {d}: {len(sfs)} files, {sum(os.path.getsize(os.path.join(path,f)) for f in sfs)/1024**3:.1f}GB")

for f in required:
    fp = os.path.join(FLUX_DIR, f)
    if os.path.exists(fp):
        print(f"  {f}: {os.path.getsize(fp)/1024**3:.1f}GB OK")
    else:
        print(f"  {f}: MISSING!")

# ====== 2. Load pipeline ======
print("\n[2/5] Loading FLUX pipeline...")
t0 = time.perf_counter()

from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(
    FLUX_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_model_cpu_offload()

load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.0f}s")

# ====== 3. Run inference ======
print(f"\n[3/5] Generating: '{PROMPT}'...")
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()

with torch.inference_mode():
    image = pipe(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=3.5,
        height=512, width=512,  # small for quick test
        generator=torch.Generator(DEV).manual_seed(42),
    ).images[0]

gen_time = time.perf_counter() - t0
vram_peak = torch.cuda.max_memory_allocated() / 1024**3
vram_now = torch.cuda.memory_allocated() / 1024**3

# ====== 4. Save & hash ======
print("\n[4/5] Saving output...")
image.save(OUTPUT)
with open(OUTPUT, "rb") as f:
    img_hash = hashlib.sha256(f.read()).hexdigest()[:16]

# ====== 5. Report ======
print(f"\n[5/5] Results:")
print(f"  {'='*40}")
print(f"  Load time:   {load_time:.0f}s")
print(f"  Gen time:    {gen_time:.1f}s ({STEPS} steps, {gen_time/STEPS:.1f}s/step)")
print(f"  VRAM peak:   {vram_peak:.1f}GB")
print(f"  VRAM now:    {vram_now:.1f}GB")
print(f"  Image hash:  {img_hash}")
print(f"  Output:      {OUTPUT}")
print(f"  {'='*40}")

# Regression baseline
baseline = {
    "prompt": PROMPT, "steps": STEPS, "hash": img_hash,
    "time_s": round(gen_time, 1), "vram_gb": round(vram_peak, 1),
    "date": time.strftime("%Y-%m-%d %H:%M"),
    "status": "PASS" if gen_time < 120 and vram_peak < 12 else "CHECK",
}
with open(r"D:\ai\output\smoke_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print(f"\n  Status: {baseline['status']}")
print(f"  Baseline saved to smoke_baseline.json")
del pipe; torch.cuda.empty_cache()
