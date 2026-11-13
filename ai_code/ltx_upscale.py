"""
LTX-Video 空间超分器 — latent 2x 上采样
ltxv-spatial-upscaler-0.9.8.safetensors
流程: 1280×704 latent → 空间超分 2x (2560×1408) → VAE decode → 原生宽2K视频
用法: python ltx_upscale.py <prompt> [steps] [frames] [base_w] [base_h]
"""
import torch, torch.nn as nn, sys, time, os
F = torch.nn.functional
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE'] = '1'
MODEL = r'D:\ai\models\LTX-Video'
OUT = r'D:\ai\output\videos'
os.makedirs(OUT, exist_ok=True)


class SpatialUpscaler(nn.Module):
    """LTX latent 空间超分器: [B,128,F,H,W] -> [B,128,F,2H,2W]"""
    def __init__(self, weights_path):
        super().__init__()
        from safetensors.torch import load_file
        sd = load_file(weights_path, device='cpu')
        self.sd = sd  # keep for lazy loading
        self.to_bf16()

    def to_bf16(self):
        self.sd = {k: v.to(dtype=torch.bfloat16) for k, v in self.sd.items()}
        return self

    def _w(self, k):
        return self.sd[k]

    def _norm(self, x, name):
        # LTX uses nn.LayerNorm(in_channels): normalize over CHANNEL dim
        # Official: hidden_states.movedim(1,-1) then LayerNorm then movedim back
        w = self._w(f'{name}.weight').to(x.device)
        b = self._w(f'{name}.bias').to(x.device)
        # x: [B, C, F, H, W] → move C to last → LayerNorm over C
        x = x.float()
        xm = x.movedim(1, -1)  # [B,F,H,W,C]
        # LayerNorm over last dim (channel)
        mean = xm.mean(dim=-1, keepdim=True)
        var = xm.var(dim=-1, keepdim=True, unbiased=False)
        out = (xm - mean) / torch.sqrt(var + 1e-5)
        out = out * w + b  # w,b shape [C] broadcast on last dim
        return out.movedim(-1, 1).to(torch.bfloat16)  # [B,C,F,H,W]

    def _conv3d(self, x, name, stride=1, padding=1):
        w = self._w(f'{name}.weight').to(x.device)
        b = self._w(f'{name}.bias').to(x.device)
        return F.conv3d(x, w, b, stride=stride, padding=padding)

    def _res_block(self, x, idx):
        # res_blocks / post_upsample_res_blocks
        prefix = f'res_blocks.{idx}' if idx >= 0 else f'post_upsample_res_blocks.{-idx-1}'
        h = self._norm(x, f'{prefix}.norm1')
        h = self._conv3d(h, f'{prefix}.conv1')
        h = F.silu(h)
        h = self._norm(h, f'{prefix}.norm2')
        h = self._conv3d(h, f'{prefix}.conv2')
        return x + h

    def forward(self, z):
        """
        z: [B, 128, F, H, W] latent
        → [B, 128, F, 2H, 2W] upscaled latent
        """
        dev = z.device
        B, C, NF, H, W = z.shape  # NF = num_frames, avoid shadowing F

        # 1. initial_conv: 128→512 (3D conv, spatial unchanged)
        x = self._conv3d(z, 'initial_conv')
        x = self._norm(x, 'initial_norm')
        x = F.silu(x)  # [B, 512, F, H, W]

        # 2. res_blocks × 4
        for i in range(4):
            x = self._res_block(x, i)

        # 3. 2D spatial upsampling: per-frame pixelshuffle 2x
        #    upsampler.0: [2048, 512, 3, 3] 2D conv
        uw = self._w('upsampler.0.weight').to(dev)  # [2048, 512, 3, 3]
        ub = self._w('upsampler.0.bias').to(dev)
        # Process each frame: [B, 512, F, H, W] → per-frame 2D
        xf = x.permute(0, 2, 1, 3, 4).reshape(B * NF, 512, H, W)  # [BF, 512, H, W]
        up = F.conv2d(xf.float(), uw.float(), ub.float(), padding=1)  # [BF, 2048, H, W]
        up = up.to(torch.bfloat16)
        # pixelshuffle 2x2: 2048 = 512 * 2 * 2
        up = F.pixel_shuffle(up, 2)  # [BF, 512, 2H, 2W]
        x = up.reshape(B, NF, 512, 2 * H, 2 * W).permute(0, 2, 1, 3, 4)  # [B, 512, F, 2H, 2W]

        # 4. post_upsample_res_blocks × 4 (3D)
        for i in range(4):
            x = self._res_block(x, -i - 1)

        # 5. final_conv: 512→128
        x = self._conv3d(x, 'final_conv')
        return x  # [B, 128, F, 2H, 2W]


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else '海浪轻轻拍打礁石, 日落金光, 电影感'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 17
    base_w = int(sys.argv[4]) if len(sys.argv) > 4 else 1280
    base_h = int(sys.argv[5]) if len(sys.argv) > 5 else 704
    print(f'[upscale] {prompt} steps={steps} frames={frames} base={base_w}x{base_h}', flush=True)

    from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5Tokenizer

    # Load components
    tok = T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer', local_files_only=True)
    te = T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder', torch_dtype=torch.bfloat16, local_files_only=True)
    vae = AutoencoderKLLTXVideo.from_pretrained(f'{MODEL}/vae', torch_dtype=torch.bfloat16, local_files_only=True)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler')
    tf = LTXVideoTransformer3DModel.from_single_file(
        f'{MODEL}/ltxv-13b-0.9.8-distilled.safetensors', torch_dtype=torch.bfloat16,
        config=r'D:\ai\ltx_config')
    pipe = LTXPipeline(transformer=tf, scheduler=sch, vae=vae, text_encoder=te, tokenizer=tok)
    pipe.enable_sequential_cpu_offload(device='cuda')

    # Load upscaler
    upscaler = SpatialUpscaler(f'{MODEL}/ltxv-spatial-upscaler-0.9.8.safetensors')
    upscaler = upscaler.to('cuda')

    # 1. Generate latent at base resolution
    print(f'[upscale] denoising at {base_w}x{base_h}...', flush=True)
    t0 = time.perf_counter()
    res = pipe(prompt=prompt, num_inference_steps=steps, num_frames=frames,
               height=base_h, width=base_w, guidance_scale=3.0,
               output_type='latent', generator=torch.Generator('cpu').manual_seed(42))
    latent = res[0] if isinstance(res, tuple) else res
    # latent could be [B,C,F,H,W] or list
    if isinstance(latent, (list, tuple)):
        latent = latent[0]
    print(f'[upscale] latent: {tuple(latent.shape)} in {time.perf_counter()-t0:.0f}s', flush=True)

    # 2. Spatial upscale latent 2x
    print(f'[upscale] upscaling latent...', flush=True)
    t1 = time.perf_counter()
    latent_hi = upscaler(latent.to('cuda', dtype=torch.bfloat16))
    print(f'[upscale] upscaled: {tuple(latent_hi.shape)} in {time.perf_counter()-t1:.0f}s', flush=True)

    # 3. VAE decode
    print(f'[upscale] VAE decode...', flush=True)
    t2 = time.perf_counter()
    with torch.inference_mode():
        # Move to VAE device (offload manages this)
        decoded = vae.decode(latent_hi.half() if hasattr(vae, 'dtype') else latent_hi).sample
        # decoded: [B, C, F, H, W] in [0,1]
    print(f'[upscale] decoded: {tuple(decoded.shape)} in {time.perf_counter()-t2:.0f}s', flush=True)

    # 4. Export video
    frames_list = [decoded[0, :, i].permute(1, 2, 0).float().cpu() for i in range(decoded.shape[2])]
    frames_list = [(f.clamp(0, 1) * 255).to(torch.uint8).numpy() for f in frames_list]
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames_list]
    fn = f'{OUT}/ltx_2k_{int(time.time())}.gif'
    imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=42, loop=0)
    print(f'[upscale] saved {fn} ({tuple(imgs[0].size)} {len(imgs)} frames)')
    print(f'[upscale] total {time.perf_counter()-t0:.0f}s VRAM peak {torch.cuda.max_memory_allocated()/1024**3:.1f}GB')


if __name__ == '__main__':
    main()
