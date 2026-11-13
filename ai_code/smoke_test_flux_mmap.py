"""
[Smoke Test] FLUX.1-dev 4-step + mmap + sequential offload
"""
import torch, time, os, mmap, hashlib
FLUX_DIR = r"D:\ai\models\FLUX.1-dev"; DEV = "cuda"
PROMPT = "A red apple on wooden table, studio lighting, 8k"
STEPS = 4

print(f"[FLUX] {PROMPT}")
t0 = time.perf_counter()

# mmap all weights
for dp, _, files in os.walk(FLUX_DIR):
    for f in files:
        if f.endswith(".safetensors"):
            fp = os.path.join(dp, f)
            fd = os.open(fp, os.O_RDONLY | os.O_BINARY)
            sz = os.fstat(fd).st_size
            mmap.mmap(fd, sz, access=mmap.ACCESS_READ)
            os.close(fd)

from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(FLUX_DIR, torch_dtype=torch.bfloat16, local_files_only=True)
pipe.enable_sequential_cpu_offload()
print(f"Loaded in {time.perf_counter()-t0:.0f}s")

torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
img = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=3.5, height=512, width=512,
           generator=torch.Generator("cpu").manual_seed(42)).images[0]
elapsed = time.perf_counter()-t0; vram = torch.cuda.max_memory_allocated()/1024**3

import os as _os; _os.makedirs(r"D:\ai\output", exist_ok=True)
OUT = r"D:\ai\output\smoke_flux.png"; img.save(OUT)
h = hashlib.sha256(open(OUT,"rb").read()).hexdigest()[:16]

print(f"Time: {elapsed:.0f}s | VRAM: {vram:.1f}GB | {STEPS} steps | hash: {h} | {OUT}")
print(f"Status: {'PASS' if elapsed < 600 else 'SLOW'} | s/step: {elapsed/STEPS:.0f}")
