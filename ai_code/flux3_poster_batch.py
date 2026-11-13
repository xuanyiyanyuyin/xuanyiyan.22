import os, sys, time, json, io
if os.path.exists(r'D:\ai\output\images\POSTER_STOP.flag'):
    print('[stop] master stop flag present, no model loaded', flush=True); sys.exit(0)
for _k in ('NO_PROXY', 'no_PROXY', 'no_proxy'):
    os.environ.pop(_k, None)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
import numpy as np

MODEL = r'D:\ai\models\FLUX.3-Klein-Base-bf16'
OUT   = r'D:\ai\output\images'
NAME = sys.argv[1] if len(sys.argv) > 1 else '暄依嫣'
DESC = sys.argv[2] if len(sys.argv) > 2 else ''
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
H = int(sys.argv[4]) if len(sys.argv) > 4 else 864
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 12
CFG = 3.5
DEV = 'cuda'

pal = []
try:
    b = json.load(io.open(r'D:\ai\cache\temp\board.json', encoding='utf-8'))
    for s in b.get('sisters', []):
        if s['n'] == NAME: pal = s.get('c', [])
except Exception: pass
desc = ''
if DESC and os.path.exists(DESC):
    desc = io.open(DESC, encoding='utf-8').read().strip()
    desc = ' '.join(desc.split())[:1200]

LAYOUT = (
 "Horizontal 16:9 professional game character asset reference sheet, clean grid layout, three independent regions, "
 "ample white space, no overlap. Top title band with Simplified Chinese '人物设定档案' and the name '%s'. "
 "Left: one large front-facing head-and-shoulders portrait, clear face looking at viewer, plus small identity card. "
 "Middle: two-row three-column grid of six expression headshots of the SAME face/hair/outfit (calm, soft smile, relaxed, "
 "focused, slightly surprised, playful wink); below a flat-lay of outfit pieces and a row of five color swatches. "
 "Right: three full-body standard views (front, exact 90-degree side, exact 180-degree back), identical size, heads aligned, "
 "feet on same baseline, head to shoes. Soft cool studio light, hyper-detailed 3D CG, professional character design sheet, 8K, "
 "consistent character across all views. "
) % NAME
PROMPT = LAYOUT + "Character appearance: " + (desc or NAME) + (". Palette: " + ", ".join(pal) if pal else "")

def log(s): print(s, flush=True)
t0 = time.time()
from diffusers import Flux3ModularPipeline
pipe = Flux3ModularPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
try: pipe._blocks.sub_blocks.pop('prompt_upsampler', None)
except Exception: pass
pipe.load_components(names=['text_encoder', 'tokenizer', 'transformer', 'vae'])
try: pipe.vae.enable_tiling()
except Exception as e: log('tiling %r' % e)
import gc
pipe.transformer.to('cpu'); gc.collect(); torch.cuda.empty_cache()
pipe.transformer.enable_group_offload(onload_device=torch.device(DEV), offload_device=torch.device('cpu'),
    offload_type='block_level', num_blocks_per_group=1, use_stream=False, record_stream=False, low_cpu_mem_usage=True)
from diffusers.modular_pipelines.components_manager import ComponentsManager
cm = ComponentsManager()
for _n in ('text_encoder', 'transformer', 'vae'):
    _m = getattr(pipe, _n, None)
    if _m is not None: cm.add(_n, _m)
pipe._components_manager = cm
cm.enable_auto_cpu_offload(device=torch.device(DEV), memory_reserve_margin='1GB')
if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
log('[gen] %s %dx%d %d steps' % (NAME, W, H, STEPS))
g = torch.Generator(device='cpu').manual_seed(42)
t1 = time.time()
out = pipe(prompt=PROMPT, height=H, width=W, num_inference_steps=STEPS, guidance_scale=CFG, generator=g)
dt = time.time() - t1
img = getattr(out, 'images', None)
if img is not None: img = img[0]
if img is None: img = out['images'][0]
arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
path = os.path.join(OUT, 'flux3_poster_%s_%dx%d.png' % (NAME, W, H))
img.save(path)
log('[done] %s gen %.1fs peak %.2fGB std %.4f path %s' % (NAME, dt, peak, arr.std(), path))