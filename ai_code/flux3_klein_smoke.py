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
PROMPT = ("A serene anime girl with long cherry-red hair fading to neon pink at the tips, "
          "faceted red crystal irises, a knowing smile, bare smooth skin, two hair-fine eyeliner "
          "strokes at each outer eye corner one deep red one glowing pink, soft rose-purple dusk "
          "background, pearl-white collar, high quality illustration")
STEPS, W, H, CFG = 50, 512, 512, 3.5
DEV = 'cuda'

def log(s): print(s, flush=True)

t0 = time.time()
log('[1] 导入 Flux3ModularPipeline ...')
from diffusers import Flux3ModularPipeline
log('[2] from_pretrained (bf16) ...')
pipe = Flux3ModularPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
log('    用时 %.1fs' % (time.time() - t0))

# 不要提示词扩写器（独立仓库未下；我的提示词已是完整场景描述）=> pop 掉该步，按原样编码
# 注意：pipe.blocks 是 deepcopy 副本，必须改实际的 pipe._blocks 才生效
try:
    pipe._blocks.sub_blocks.pop('prompt_upsampler', None)
    log('[2b] 已 pop prompt_upsampler 步（剩余步: %s）' % list(pipe._blocks.sub_blocks.keys()))
except Exception as e:
    log('[2b] pop upsampler 失败: %r' % e)

log('[3] load_components（只加载 text_encoder/tokenizer/transformer/vae）...')
pipe.load_components(names=['text_encoder', 'tokenizer', 'transformer', 'vae'])
log('    用时 %.1fs' % (time.time() - t0))

# transformer 11GB bf16 > 空闲显存 => 块级流式（每次只驻留 1 块）
log('[4] transformer 块级流式 group_offload ...')
import gc
pipe.transformer.to('cpu'); gc.collect(); torch.cuda.empty_cache()   # 先确保在 CPU，避免 pin_memory 时撞显存
pipe.transformer.enable_group_offload(
    onload_device=torch.device(DEV), offload_device=torch.device('cpu'),
    offload_type='block_level', num_blocks_per_group=1,
    use_stream=False, record_stream=False, low_cpu_mem_usage=True)   # 跳过 pin_memory（cuMemHostAlloc OOM 的解法）
log('    transformer group_offload OK')
# 其余组件（text_encoder 8.9GB / vae）按显存自动 CPU<->GPU
log('[5] 其余组件 auto_cpu_offload ...')
from diffusers.modular_pipelines.components_manager import ComponentsManager
cm = ComponentsManager()
for _n in ('text_encoder', 'transformer', 'vae'):
    _m = getattr(pipe, _n, None)
    if _m is not None:
        cm.add(_n, _m)
pipe._components_manager = cm
cm.enable_auto_cpu_offload(device=torch.device(DEV), memory_reserve_margin='1GB')
log('    auto_cpu_offload OK（注册 %d 个组件）' % len(cm.components))

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
log('[6] 生成 %dx%d / %d步 / CFG%.1f ...' % (W, H, STEPS, CFG))
g = torch.Generator(device='cpu').manual_seed(42)
t1 = time.time()
out = pipe(prompt=PROMPT, height=H, width=W, num_inference_steps=STEPS,
           guidance_scale=CFG, generator=g)
dt = time.time() - t1
img = getattr(out, 'images', None)
if img is not None: img = img[0]
if img is None:
    try: img = out['images'][0]
    except Exception:
        log('    输出类型 %s' % type(out)); raise SystemExit('没拿到 image')
arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
path = os.path.join(OUT, 'flux3_klein_base_smoke_512.png')
img.save(path)
log('[7] 完成: 生成用时 %.1fs | 显存峰值 %.2fGB | mean=%.4f std=%.4f' % (dt, peak, arr.mean(), arr.std()))
log('    塌缩判据 std<0.02: 本图 std=%.4f => %s' % (arr.std(), 'OK' if arr.std() >= 0.02 else '疑似塌缩'))
log('    图: %s' % path)
log('总耗时 %.1fs' % (time.time() - t0))