import torch, sys, time
sys.path.insert(0, r'D:\ai')
from diffusers import AutoencoderKL
import numpy as np
MODEL=r'D:\ai\models\FLUX.1-dev'
# 用同一个 latent, 分别 tiling / 不tiling decode, 算差异
vae=AutoencoderKL.from_pretrained(MODEL+'/vae', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
torch.manual_seed(7)
W,H=1024,1024   # 测试尺寸, 快速
z=torch.randn(1,16,H//8,W//8,dtype=torch.bfloat16,device='cuda')

# 不 tiling
img_ref=None
with torch.no_grad():
    img_ref=vae.decode(z).sample.float().cpu()
torch.cuda.empty_cache()

# tiling
vae.enable_tiling()
with torch.no_grad():
    img_tile=vae.decode(z).sample.float().cpu()
torch.cuda.empty_cache()

# 差异统计
d=(img_ref-img_tile).abs()
print(f'尺寸 {W}x{H}')
print(f'  tiling vs 非tiling 最大差: {d.max().item():.3f}')
print(f'  平均差: {d.mean().item():.5f}')
print(f'  差异>0.02 像素占比: {(d>0.02).mean().item()*100:.2f}%')
print('  (>0.02差异占比越小 越无缝隙)')
vae.enable_tiling()
