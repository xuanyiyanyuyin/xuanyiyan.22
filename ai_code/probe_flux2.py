import torch, sys, time
sys.path.insert(0, r'D:\ai')
MODEL=r'D:\ai\models\FLUX.2-dev'
t0=time.perf_counter()
from diffusers import Flux2Pipeline
print('加载 Flux2Pipeline (CPU)...', flush=True)
pipe=Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完成 {time.perf_counter()-t0:.0f}s', flush=True)
t=pipe.transformer
print('transformer 类型:', type(t).__name__, flush=True)
# 块数量
for attr in ['transformer_blocks','single_transformer_blocks','blocks','layers']:
    if hasattr(t,attr):
        print(f'  {attr}: {len(getattr(t,attr))}', flush=True)
# 判断 block 类型
import torch.nn as nn
for sub in ['transformer_blocks','blocks']:
    if hasattr(t,sub) and len(getattr(t,sub))>0:
        b=getattr(t,sub)[0]
        print(f'  示例块 {sub}[0]: {type(b).__name__}', flush=True)
        nparam=sum(p.numel()*p.element_size() for p in b.parameters())/1024**3
        print(f'      权重≈{nparam:.2f}G', flush=True)
        break
# text_encoder
te=pipe.text_encoder
print('text_encoder:', type(te).__name__, flush=True)
# 全变压器总参数
tot=sum(p.numel()*p.element_size() for p in t.parameters())/1024**3
print(f'transformer 总权重 ≈ {tot:.1f}G (BF16)', flush=True)
