"""
LTX-Video 13B-distilled 魔改生视频 — 12GB显存流式版
策略: diffusers官方计算 + 逐块CPU卸载; transformer 13B via from_single_file
超分: 1280×704 latent → 空间超分2x → VAE decode → 2560×1408
用法: D:\ai\python\python.exe gen_video.py "english prompt" [steps] [frames]
"""
import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE']='1'
MODEL=r'D:\ai\models\LTX-Video'
OUT=r'D:\ai\output\videos'
os.makedirs(OUT, exist_ok=True)

_pipe=None
_upscaler=None

def get_pipe():
    global _pipe
    if _pipe is not None: return _pipe
    from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    t0=time.perf_counter()
    # 能跑版本: 分片目录 + from_pretrained(low_cpu_mem_usage) — 绕开 from_single_file 偶发段错误 (照122B)
    # 分片由 _split_pread.py 从 single-file 转换 (pread读+重命名), 流式加载不占内存峰值
    tf=LTXVideoTransformer3DModel.from_pretrained(
        f'{MODEL}/transformer_13b_shards', torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True)
    print(f'[ltx] transformer(13B) loaded {time.perf_counter()-t0:.0f}s',flush=True)
    import gc
    gc.collect()
    tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
    te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
    vae=AutoencoderKLLTXVideo.from_pretrained(f'{MODEL}/vae_1024',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler')
    print(f'[ltx] components loaded {time.perf_counter()-t0:.0f}s',flush=True)
    _pipe=LTXPipeline(transformer=tf,scheduler=sch,vae=vae,text_encoder=te,tokenizer=tok)
    _pipe.enable_sequential_cpu_offload(device='cuda')
    print(f'[ltx] offload ready {time.perf_counter()-t0:.0f}s',flush=True)
    return _pipe

def free_pipe():
    global _pipe
    if _pipe is not None:
        del _pipe; _pipe=None
        torch.cuda.empty_cache()
        print('[ltx] released',flush=True)

def get_upscaler():
    global _upscaler
    if _upscaler is not None: return _upscaler
    from ltx_upscale import SpatialUpscaler
    _upscaler=SpatialUpscaler(f'{MODEL}/ltxv-spatial-upscaler-0.9.8.safetensors').to('cuda')
    return _upscaler

def generate_video(prompt, steps=50, frames=33, base_w=2560, base_h=1408,
                   do_upscale=False, seed=42, target=(2560,1408), input_image=None):
    """生成视频 → MP4
    input_image: 图生视频 — PIL.Image 或图片路径，作为首帧条件（i2v）
    do_upscale=True: latent超分 (需内存充裕)
    do_upscale=False: 原生2K生成 2560×1408 (推荐, 禁止硬放大)
    返回: (路径, 分辨率, 帧数)"""
    t0=time.perf_counter()
    pipe=get_pipe()
    upscaler=get_upscaler() if do_upscale else None
    gen=torch.Generator('cpu').manual_seed(seed)
    # 图生视频: 加载首帧图片 (diffusers LTXPipeline 支持 image 参数)
    if input_image is not None:
        from PIL import Image as _PIL
        if isinstance(input_image, str):
            input_image = _PIL.open(input_image).convert('RGB')
        elif hasattr(input_image, 'convert'):
            input_image = input_image.convert('RGB')
        input_image = input_image.resize((base_w, base_h), _PIL.LANCZOS)
        print(f'[ltx-i2v] 图生视频: 首帧 {input_image.size}', flush=True)
    res=pipe(prompt=prompt, image=input_image,
             num_inference_steps=steps,num_frames=frames,
             height=base_h,width=base_w,guidance_scale=3.0,
             output_type='latent' if upscaler else 'pil',
             generator=gen)
    if upscaler:
        # latent是打包格式 [B,S,D] → unpack为 [B,C,F,H,W]
        latent=res.frames
        if isinstance(latent,(list,tuple)): latent=latent[0]
        lat_h=base_h//32; lat_w=base_w//32  # LTX VAE 32x空间压缩
        latent5=pipe._unpack_latents(latent, frames, lat_h, lat_w)
        print(f'[ltx] latent {tuple(latent5.shape)}, upscaling...',flush=True)
        latent_hi=upscaler(latent5.to('cuda',dtype=torch.bfloat16))  # [B,C,F,2H,2W]
        # 重新打包为 [B,S,D] 供 vae.decode
        latent_packed=pipe._pack_latents(latent_hi)
        with torch.inference_mode():
            decoded=pipe.vae.decode(latent_packed.to(torch.bfloat16), return_dict=False)[0]
        fr=[decoded[0,:,i].permute(1,2,0).float().cpu().clamp(0,1) for i in range(decoded.shape[2])]
        from PIL import Image
        fr=[Image.fromarray((f*255).to(torch.uint8).numpy()) for f in fr]
    else:
        fr=res.frames[0]
        from PIL import Image
        fr=[Image.fromarray(f) if not isinstance(f,Image.Image) else f for f in fr]
        # 像素放大到 target (原生2K)
        if fr[0].size != target:
            print(f'[ltx] upscaling pixels {fr[0].size} -> {target}',flush=True)
            fr=[f.resize(target, Image.LANCZOS) for f in fr]
    ts_=int(time.time())
    fn=f'{OUT}/ltx_{ts_}.mp4'
    save_video_mp4(fr, fn, fps=24)
    print(f'[ltx] saved {fn} ({fr[0].size}, {len(fr)}f) {time.perf_counter()-t0:.0f}s VRAM {torch.cuda.max_memory_allocated()/1024**3:.1f}GB',flush=True)
    return fn, fr[0].size, len(fr)

def save_video_mp4(frames, out_path, fps=12, crf=16):
    """ffmpeg管道输出MP4 — H.264 CRF16高画质, NVENC优先
    frames: list[PIL.Image] 或 list[np.ndarray]
    crf: 16≈近无损, 18=标准, 越小画质越高
    """
    import numpy as np
    import imageio_ffmpeg, subprocess
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    w,h=frames[0].size if hasattr(frames[0],'size') else frames[0].shape[1::-1]
    # 尝试 NVENC, 失败回退 libx264
    enc='h264_nvenc'
    # 检测NVENC可用性
    probe=subprocess.run([exe,'-encoders'],capture_output=True,text=True)
    if 'h264_nvenc' not in probe.stdout:
        enc='libx264'
    cmd=[exe,'-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{w}x{h}','-r',str(fps),'-i','-',
         '-c:v',enc,'-crf',str(crf),'-preset','p7' if enc=='h264_nvenc' else 'slow',
         '-pix_fmt','yuv420p','-movflags','+faststart',
         out_path]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for fr in frames:
        arr=np.array(fr.convert('RGB') if hasattr(fr,'convert') else fr,dtype=np.uint8)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode!=0:
        raise RuntimeError(f'ffmpeg failed: {out_path} rc={proc.returncode}')

def main():
    prompt=sys.argv[1] if len(sys.argv)>1 else 'waves gently hitting rocks at sunset, cinematic'
    steps=int(sys.argv[2]) if len(sys.argv)>2 else 50
    frames=int(sys.argv[3]) if len(sys.argv)>3 else 33
    input_image=sys.argv[4] if len(sys.argv)>4 else None  # 图生视频: 第4参 = 图片路径
    fn,size,n=generate_video(prompt,steps=steps,frames=frames,input_image=input_image)
    print(f'[ltx] done -> {fn} {size} {n}f',flush=True)
    free_pipe()

if __name__=='__main__':
    main()
