import sys, time, torch, faulthandler
faulthandler.enable()
sys.path.insert(0, r'D:\ai')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from diffusers import Flux2Pipeline
print('加载 pipeline...', flush=True)
t0=time.perf_counter()
pipe=Flux2Pipeline.from_pretrained(r'D:\ai\models\FLUX.2-dev', torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完 {time.perf_counter()-t0:.0f}s', flush=True)
print('encode_prompt(device=cpu)...', flush=True)
pe, tid = pipe.encode_prompt('a cute cat', device='cpu', max_sequence_length=512, prompt_embeds=None, num_images_per_prompt=1)
print(f'encode完成 {time.perf_counter()-t0:.0f}s pe={tuple(pe.shape)}', flush=True)
