import sys, time
sys.path.insert(0, r'D:\ai')
import torch, psutil
from diffusers import Flux2Pipeline
import flux2_3tier as f2
t0=time.perf_counter()
print('加载 pipeline...', flush=True)
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完 {time.perf_counter()-t0:.0f}s', flush=True)
print('开始流式编码(40层逐层上GPU)...', flush=True)
pe, tid = f2.encode_prompt_streaming(pipe, 'a cute orange cat on white table', max_seq=512)
print(f'编码完成 {time.perf_counter()-t0:.0f}s', flush=True)
print(f'pe={tuple(pe.shape)} tid={tuple(tid.shape)}', flush=True)
print('done', flush=True)
