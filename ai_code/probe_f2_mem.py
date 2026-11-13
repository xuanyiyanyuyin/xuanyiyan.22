# -*- coding: utf-8 -*-
import sys, time, psutil, os, torch
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE']='1'
def mem(tag):
    print(f'  [{tag}] RAM avail={psutil.virtual_memory().available/1024**3:.1f}G', flush=True)
t0=time.perf_counter()
mem('start')
from diffusers import Flux2Pipeline
pipe=Flux2Pipeline.from_pretrained(r'D:\ai\models\FLUX.2-dev', torch_dtype=torch.bfloat16, local_files_only=True)
mem(f'load完 {time.perf_counter()-t0:.0f}s')
te=pipe.text_encoder
nparam=sum(p.numel() for p in te.parameters())
ram_bytes=sum(p.numel()*p.element_size() for p in te.parameters() if p.device.type=='cpu')
print(f'text_encoder 参数 {nparam/1e9:.1f}B, CPU张量占用 ~{ram_bytes/1024**3:.1f}G', flush=True)
mem('check完')
