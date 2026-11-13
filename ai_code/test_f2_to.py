import sys, time, torch, psutil
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
lm=pipe.text_encoder.model.language_model
# 测单个 layer .to('cuda') 耗时
t=time.perf_counter()
lm.layers[0].to('cuda')
torch.cuda.synchronize()
print(f'layer0.to(cuda) 首{time.perf_counter()-t:.1f}s', flush=True)
t=time.perf_counter()
lm.layers[1].to('cuda')
torch.cuda.synchronize()
print(f'layer1.to(cuda) {time.perf_counter()-t:.1f}s', flush=True)
t=time.perf_counter()
lm.layers[2].to('cuda')
torch.cuda.synchronize()
print(f'layer2.to(cuda) {time.perf_counter()-t:.1f}s', flush=True)
print('GPU alloc:', round(torch.cuda.memory_allocated()/1024**3,2),'G', flush=True)
print('RAM avail:', round(psutil.virtual_memory().available/1024**3,1),'G', flush=True)
