import os, sys, time, json, io
if os.path.exists(r'D:\ai\output\images\POSTER_STOP.flag'):
    print('[stop] master stop flag present, no model loaded', flush=True); sys.exit(0)
for _k in ('NO_PROXY', 'no_PROXY', 'no_proxy'):
    os.environ.pop(_k, None)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_DISABLE_XET'] = '1'
import torch
import numpy as np
from PIL import Image

MODEL = r'D:\ai\models\FLUX.3-Klein-Base-bf16'
OUT   = r'D:\ai\output\images'
NAME = sys.argv[1] if len(sys.argv) > 1 else '暄依嫣'
DESC = sys.argv[2] if len(sys.argv) > 2 else ''
DEV = 'cuda'
TW, TH = 1408, 800          # tile size (VRAM ~ like 1536x864)
FX, FY = 128, 80           # feather margin (half overlap)
BASE_STEPS, TILE_STEPS = 12, 8

pal = []
try:
    b = json.load(io.open(r'D:\ai\cache\temp\board.json', encoding='utf-8'))
    for s in b.get('sisters', []):
        if s['n'] == NAME: pal = s.get('c', [])
except Exception: pass
desc = ''
if DESC and os.path.exists(DESC):
    desc = ' '.join(io.open(DESC, encoding='utf-8').read().split())[:1200]
LAYOUT = ("Horizontal 16:9 professional game character asset reference sheet, three independent regions, ample white space. "
 "Top band Simplified Chinese '人物设定档案' and name '%s'. Left large front portrait + identity card. Middle 2x3 six-expression "
 "grid of same face + outfit flat-lay + five color swatches. Right three full-body views front/side/back aligned same baseline. "
 "Soft cool studio light, hyper-detailed 3D CG, 8K, consistent character. ") % NAME
PROMPT = LAYOUT + "Character: " + (desc or NAME) + (". Palette: " + ", ".join(pal) if pal else "")

def log(s): print(s, flush=True)

def load_pipe():
    from diffusers import Flux3ModularPipeline
    p = Flux3ModularPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    try: p._blocks.sub_blocks.pop('prompt_upsampler', None)
    except Exception: pass
    p.load_components(names=['text_encoder', 'tokenizer', 'transformer', 'vae'])
    try: p.vae.enable_tiling()
    except Exception: pass
    import gc
    p.transformer.to('cpu'); gc.collect(); torch.cuda.empty_cache()
    p.transformer.enable_group_offload(onload_device=torch.device(DEV), offload_device=torch.device('cpu'),
        offload_type='block_level', num_blocks_per_group=1, use_stream=False, record_stream=False, low_cpu_mem_usage=True)
    from diffusers.modular_pipelines.components_manager import ComponentsManager
    cm = ComponentsManager()
    for n in ('text_encoder', 'transformer', 'vae'):
        m = getattr(p, n, None)
        if m is not None: cm.add(n, m)
    p._components_manager = cm
    cm.enable_auto_cpu_offload(device=torch.device(DEV), memory_reserve_margin='1GB')
    return p

def gen(p, prompt, w, h, steps, image=None, seed=42):
    g = torch.Generator(device='cpu').manual_seed(seed)
    kw = dict(prompt=prompt, height=h, width=w, num_inference_steps=steps, guidance_scale=3.5, generator=g)
    if image is not None: kw['image'] = image
    out = p(**kw)
    im = getattr(out, 'images', None)
    if im is not None: im = im[0]
    if im is None: im = out['images'][0]
    return im.convert('RGB')

t0 = time.time()
pipe = load_pipe()
base_path = os.path.join(OUT, 'flux3_poster_%s_1536x864.png' % NAME)
if os.path.exists(base_path):
    base = Image.open(base_path).convert('RGB')
    log('[base] reuse %s' % base_path)
else:
    base = gen(pipe, PROMPT, 1536, 864, BASE_STEPS)
    base.save(base_path); log('[base] generated')
up = base.resize((2560, 1440), Image.LANCZOS)

acc = np.zeros((1440, 2560, 3), np.float64); wacc = np.zeros((1440, 2560), np.float64)
xs = [0, 2560 - TW]; ys = [0, 1440 - TH]
def feather(h, w, fy, fx):
    wy = np.ones(h); wx = np.ones(w)
    ry = np.linspace(0, 1, fy); rx = np.linspace(0, 1, fx)
    wy[:fy] = ry; wy[-fy:] = ry[::-1]; wx[:fx] = rx; wx[-fx:] = rx[::-1]
    return wy[:, None] * wx[None, :]
n = 0
for y in ys:
    for x in xs:
        n += 1
        tile = up.crop((x, y, x + TW, y + TH))
        r = gen(pipe, PROMPT, TW, TH, TILE_STEPS, image=tile, seed=42 + n)
        ra = np.asarray(r, np.float64) / 255.0
        w = feather(TH, TW, FY, FX)
        acc[y:y+TH, x:x+TW] += ra * w[:, :, None]; wacc[y:y+TH, x:x+TW] += w
        log('[tile %d/4] (%d,%d) done %.0fs' % (n, x, y, time.time() - t0))
final = (acc / np.maximum(wacc, 1e-6)[:, :, None] * 255).clip(0, 255).astype(np.uint8)
path = os.path.join(OUT, 'flux3_poster_%s_2560x1440_native.png' % NAME)
Image.fromarray(final).save(path)
log('[done] %s native2560 path %s total %.0fs' % (NAME, path, time.time() - t0))