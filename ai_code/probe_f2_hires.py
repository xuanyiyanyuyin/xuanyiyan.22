import sys, time, torch
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
import faulthandler; faulthandler.enable()
import numpy as np
t0 = time.perf_counter()

W, H = int(sys.argv[1]) if len(sys.argv) > 1 else 2560, int(sys.argv[2]) if len(sys.argv) > 2 else 1408
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # resident_d
NCHK = int(sys.argv[4]) if len(sys.argv) > 4 else 1  # 只跑 N 步探测

print(f'[probe] 分辨率 {W}x{H}  resident_d={RES}  nchk={NCHK}', flush=True)
print('[probe] 加载 embeds...', flush=True)
ck = torch.load(r'D:\ai\_cached_flux2_embeds_test.pt', map_location='cpu', weights_only=True)
prompt_embeds = ck['embeds'].to('cuda').bfloat16()

from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    low_cpu_mem_usage=True, text_encoder=None)
print(f'[probe] pipe加载 {time.perf_counter()-t0:.0f}s', flush=True)
r = f2._patch_f2(pipe.transformer, resident_d=RES)
print(f'[probe] patch resident={r}', flush=True)

text_ids = pipe._prepare_text_ids(prompt_embeds).to('cuda')
nch = pipe.transformer.config.in_channels // 4
latents, latent_ids = pipe.prepare_latents(1, nch, W, H, prompt_embeds.dtype, 'cuda',
    torch.Generator('cuda').manual_seed(1), None)
print(f'[probe] latents {tuple(latents.shape)} img_seq={(W//16)*(H//16)}', flush=True)

img_seq = (W // 16) * (H // 16)
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
NSTEPS = NCHK
mu = compute_empirical_mu(image_seq_len=img_seq, num_steps=NSTEPS)
ts, _ = retrieve_timesteps(pipe.scheduler, NSTEPS, 'cuda',
    sigmas=np.linspace(1.0, 1.0/NSTEPS, NSTEPS), mu=mu)
guidance = torch.full([1], 4.0, device='cuda', dtype=torch.float32).expand(latents.shape[0])

def peak():
    return torch.cuda.max_memory_allocated()/1e9, torch.cuda.memory_allocated()/1e9

torch.cuda.reset_peak_memory_stats()
for i, t in enumerate(ts):
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    npred = pipe.transformer(hidden_states=latents.to(pipe.transformer.dtype),
        timestep=timestep/1000, guidance=guidance,
        encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_ids,
        joint_attention_kwargs=None, return_dict=False)[0]
    npred = npred[:, :latents.size(1)]
    latents = pipe.scheduler.step(npred, t, latents, return_dict=False)[0]
    del npred; torch.cuda.empty_cache()
    pk, cur = peak()
    print(f'[probe] step{i} t+{time.perf_counter()-t0:.0f}s  peak={pk:.2f}GB cur={cur:.2f}GB', flush=True)

torch.cuda.synchronize()
pk, cur = peak()
print(f'[probe] DONE {W}x{H} resident={RES} steps={NSTEPS}  PEAK={pk:.2f}GB  cur={cur:.2f}GB', flush=True)
print(f'[probe] 显存余量: {12.2-pk:.2f}GB', flush=True)
