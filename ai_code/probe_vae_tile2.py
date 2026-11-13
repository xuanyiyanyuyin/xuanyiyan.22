
import torch, sys, time
sys.path.insert(0, r'D:\ai')
MODEL=r'D:\ai\models\FLUX.1-dev'
RES=[]
def run(tile, W, H):
    from diffusers import AutoencoderKL
    torch.cuda.empty_cache()
    pipe=AutoencoderKL.from_pretrained(MODEL+'/vae', torch_dtype=torch.bfloat16, local_files_only=True)
    pipe=pipe.to('cuda')
    if tile: pipe.enable_tiling()
    t0=time.perf_counter()
    x=torch.randn(1,3,H,W,dtype=torch.bfloat16,device='cuda')
    with torch.no_grad():
        z=pipe.encode(x).latent_dist.sample()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/1024**3
    torch.cuda.reset_peak_memory_stats()
    RES.append(f"tile={tile} {W}x{H}: peak={peak:.2f}GB, latent={tuple(z.shape)}, {time.perf_counter()-t0:.1f}s")
    del pipe,z,x; torch.cuda.empty_cache()
run(False,3840,2160)
run(True,3840,2160)
open(r'D:\ai\_vt_result.txt','w',encoding='utf-8').write('\n'.join(RES))
