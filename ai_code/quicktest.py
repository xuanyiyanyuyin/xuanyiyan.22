"""
快速端到端测试: FLUX 256小图 → VL-30B点评
Expected: ~3-5 min total
"""
import torch, time, sys, os
sys.path.insert(0, r'D:\ai')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Step 1: FLUX 256×256 small image ──
print('='*50)
print('Step 1: FLUX 256×256 小图 (预计 ~30s)')
print('='*50)
t0 = time.perf_counter()

from diffusers import FluxPipeline
MODEL = r'D:\ai\models\FLUX.1-dev'
OUT = r'D:\ai\output\images'
os.makedirs(OUT, exist_ok=True)

prompt = "a cute orange cat, simple illustration, flat colors"
print(f'Loading FLUX...')
pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
pipe.enable_sequential_cpu_offload(device='cuda')
print(f'Generating 256×256...')

img = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=3.5,
           height=256, width=256, max_sequence_length=256,
           generator=torch.Generator('cpu').manual_seed(42)).images[0]

test_path = f'{OUT}/_quicktest.png'
img.save(test_path)
print(f'Saved: {test_path} ({img.size})')
del pipe
torch.cuda.empty_cache()
t_flux = time.perf_counter() - t0
print(f'FLUX done in {t_flux:.0f}s')

# ── Step 2: VL-30B 点评 ──
print()
print('='*50)
print('Step 2: VL-30B 看图点评 (预计 ~2-3min)')
print('='*50)
t1 = time.perf_counter()

from vl_scorer import get_scorer
scorer = get_scorer()
result = scorer.score_image(test_path)

print(f'VL done in {time.perf_counter()-t1:.0f}s')
print()
print('=== RESULT ===')
commentary = result.get('commentary', '')
if commentary:
    try:
        print(commentary)
    except:
        print('[commentary omitted - encoding]')
print(f"Overall: {result.get('overall','?')}/10")
print(f"Passed: {result.get('passed','?')}")
print(f"Source: {result.get('source','?')}")
print(f"Total time: {time.perf_counter()-t0:.0f}s")
