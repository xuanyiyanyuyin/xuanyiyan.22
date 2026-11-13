import sys, time
sys.path.insert(0, r'D:\ai')
import torch, psutil
t0=time.perf_counter()
from diffusers import Flux2Pipeline
print('加载 pipeline...', flush=True)
pipe=Flux2Pipeline.from_pretrained(r'D:\ai\models\FLUX.2-dev', torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完 {time.perf_counter()-t0:.0f}s, avail={psutil.virtual_memory().available/1024**3:.1f}G', flush=True)
# 只编码, 不跑transformer
print('encode_prompt...', flush=True)
try:
    pe, tid = pipe.encode_prompt(prompt='a cute cat', device='cpu', max_sequence_length=512, prompt_embeds=None, num_images_per_prompt=1)
    print(f'编码OK pe={pe.shape} tid={tid.shape} avail={psutil.virtual_memory().available/1024**3:.1f}G', flush=True)
except Exception as e:
    import traceback
    print('编码失败:', e, flush=True)
    traceback.print_exc()
# 释放 text_encoder
del pipe.text_encoder
torch.cuda.empty_cache()
print(f'释放后 avail={psutil.virtual_memory().available/1024**3:.1f}G', flush=True)
