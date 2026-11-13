import torch, sys, time
sys.path.insert(0, r'D:\ai')
MODEL=r'D:\ai\models\FLUX.1-dev'
def run(tile, W, H):
    from diffusers import AutoencoderKL
    import torch
    torch.cuda.empty_cache()
    pipe=AutoencoderKL.from_pretrained(MODEL+'/vae', torch_dtype=torch.bfloat16, local_files_only=True)
    pipe=pipe.to('cuda')
    if tile:
        pipe.enable_tiling()
    print(f'--- tile={tile} 尺寸{W}x{H} ---', flush=True)
    print('tile_sample_min_size=%s latent_min=%s overlap=%s'%(pipe.tile_sample_min_size,pipe.tile_latent_min_size,pipe.tile_overlap_factor),flush=True)
    x=torch.randn(1,3,H,W,dtype=torch.bfloat16,device='cuda')
    t0=time.perf_counter()
    with torch.no_grad():
        z=pipe.encode(x).latent_dist.sample()
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()/1024**3
    print(f'编码完成 {time.perf_counter()-t0:.1f}s, latent=x{z.shape}, 峰值显存={peak:.2f}GB', flush=True)
    del pipe,x,z
    torch.cuda.empty_cache()

for tile in [False, True]:
    run(tile, 3840, 2160)   # 4K
