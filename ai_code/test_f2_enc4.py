import sys, time, torch
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
import psutil
t0=time.perf_counter()
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载 {time.perf_counter()-t0:.0f}s', flush=True)
pe, tid = f2.encode_prompt_streaming(pipe, 'a cute orange cat on white table', max_seq=512)
print(f'编码完成 {time.perf_counter()-t0:.0f}s pe={tuple(pe.shape)}', flush=True)
