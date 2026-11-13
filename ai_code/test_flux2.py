import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
MODEL=r'D:\ai\models\FLUX.2-dev'
t0=time.perf_counter()
from diffusers import Flux2Pipeline
print('加载 Flux2Pipeline...', flush=True)
pipe=Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完成 {time.perf_counter()-t0:.0f}s, 启offload', flush=True)
pipe.enable_sequential_cpu_offload()
print('开始生成(512x512, 8步)...', flush=True)
res=pipe(prompt='a cute orange cat on a white table, photorealistic',
         num_inference_steps=8, height=512, width=512, guidance_scale=3.5, max_sequence_length=512)
img=res.images[0]
fn=r'D:\ai\output\images\flux2_test.png'
img.save(fn)
print(f'FLUX2_OUT {fn} {time.perf_counter()-t0:.0f}s', flush=True)
