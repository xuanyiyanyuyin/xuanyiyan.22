import os, sys, time
for _k in ('NO_PROXY', 'no_PROXY', 'no_proxy'):
    os.environ.pop(_k, None)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
import numpy as np

MODEL = r'D:\ai\models\FLUX.3-Klein-Base-bf16'
OUT   = r'D:\ai\output\images'
os.makedirs(OUT, exist_ok=True)
W = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
H = int(sys.argv[2]) if len(sys.argv) > 2 else 864
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
TAG = sys.argv[4] if len(sys.argv) > 4 else 'xuanyiyan'
CFG = 3.5
DEV = 'cuda'

PROMPT = (
 "Horizontal 16:9 professional game character asset reference sheet, clean grid layout with three independent "
 "regions and ample white space, no overlap. Top title band with Simplified Chinese text '人物设定档案' and the "
 "name '暄依嫣'. Left region: one large front-facing head-and-shoulders portrait of an elegant gentle young woman "
 "with long soft pink-orange hair and warm gold accents, clear face looking at viewer, plus a small identity card. "
 "Middle region: a two-row three-column grid of six expression headshots of the SAME face and hairstyle (calm, soft "
 "smile, relaxed, focused, slightly surprised, playful wink), below them a flat-lay of her outfit pieces and a row of "
 "five color swatches (blush pink, peach, coral orange, soft purple, gold). Right region: three full-body standard "
 "views side by side (front, exact 90-degree side, exact 180-degree back), identical size, heads aligned at the same "
 "height and feet on the same baseline, showing the full outfit from head to shoes. Soft cool studio lighting, "
 "hyper-detailed 3D CG render, professional character design sheet, 8K detail, consistent character across all views."
)

def log(s): print(s, flush=True)
t0 = time.time()
log('[1] Flux3ModularPipeline bf16 ...')
from diffusers import Flux3ModularPipeline
pipe = Flux3ModularPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
try:
    pipe._blocks.sub_blocks.pop('prompt_upsampler', None)
except Exception as e:
    log('pop upsampler: %r' % e)
pipe.load_components(names=['text_encoder', 'tokenizer', 'transformer', 'vae'])
try:
    pipe.vae.enable_tiling()
    log('[1b] vae tiling on (decode large latent in tiles)')
except Exception as e:
    log('vae tiling: %r' % e)
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
log('[2] generate %dx%d / %d steps ...' % (W, H, STEPS))
g = torch.Generator(device='cpu').manual_seed(42)
t1 = time.time()
out = pipe(prompt=PROMPT, height=H, width=W, num_inference_steps=STEPS, guidance_scale=CFG, generator=g)
dt = time.time() - t1
img = getattr(out, 'images', None)
if img is not None: img = img[0]
if img is None: img = out['images'][0]
arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
path = os.path.join(OUT, 'flux3_poster_%s_%dx%d.png' % (TAG, W, H))
img.save(path)
log('[3] done gen %.1fs (%.1fs/step) peak %.2fGB mean %.4f std %.4f' % (dt, dt / STEPS, peak, arr.mean(), arr.std()))
log('path %s' % path)
log('total %.1fs' % (time.time() - t0))