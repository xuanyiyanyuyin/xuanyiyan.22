import torch, sys, time
sys.path.insert(0, r'D:\ai')
from diffusers import AutoencoderKL
MODEL=r'D:\ai\models\FLUX.1-dev'
def test(tile, W,H):
    torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(MODEL+'/vae', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    if tile: vae.enable_tiling()
    z=torch.randn(1,16,H//8,W//8,dtype=torch.bfloat16,device='cuda')
    t0=time.perf_counter()
    with torch.no_grad():
        img=vae.decode(z).sample
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/1024**3
    print(f'tile={tile} {W}x{H}: 成功 {time.perf_counter()-t0:.1f}s, 峰值={peak:.2f}G',flush=True)
    del vae,z,img; torch.cuda.empty_cache()
test(True,3840,2160)   # 4K + tiling
