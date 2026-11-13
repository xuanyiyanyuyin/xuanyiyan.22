import torch, sys, time
sys.path.insert(0, r'D:\ai')
from diffusers import AutoencoderKL
MODEL=r'D:\ai\models\FLUX.1-dev'
def test(W,H):
    torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(MODEL+'/vae', torch_dtype=torch.bfloat16, local_files_only=True).to('cuda')
    print(f'--- VAE decode {W}x{H} ---',flush=True)
    # 构造 latent (FLUX latent 16通道, /8下采样)
    lh, lw = H//8, W//8
    z=torch.randn(1,16,lh,lw,dtype=torch.bfloat16,device='cuda')
    t0=time.perf_counter()
    try:
        with torch.no_grad():
            img=vae.decode(z).sample
        torch.cuda.synchronize()
        print(f'  成功! {time.perf_counter()-t0:.1f}s, img={tuple(img.shape)}, 峰值={torch.cuda.max_memory_allocated()/1024**3:.2f}G',flush=True)
    except Exception as e:
        print(f'  错误: {type(e).__name__}: {e}',flush=True)
    del vae, z; torch.cuda.empty_cache()
test(2560,1408)
test(3840,2160)
