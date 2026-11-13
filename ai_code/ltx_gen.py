"""LTX 视频生成 — 三级调度手写引擎 (用户方案落地)
T5 串行隔离: 先加载→encode→卸载
transformer: 三级调度 (双盘+环形缓冲+GPU prefetch), 手写 forward (已验证与官方一致)
VAE: 最后加载 decode
"""
import sys, io, os, time, math, gc
sys.path.insert(0, r'D:\ai')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
os.environ['HF_HUB_OFFLINE']='1'
import torch
import numpy as np
from ltx_sched import transformer_forward_sched, compute_rope

MODEL=r'D:\ai\models\LTX-Video'
OUT=r'D:\ai\output\videos'
os.makedirs(OUT, exist_ok=True)
DEV='cuda'

# ═══ latent 打包/解包 (照 LTXPipeline) ═══
def pack_latents(latents, patch_size=1, patch_size_t=1):
    B,C,F,H,W=latents.shape
    pf=F//patch_size_t; ph=H//patch_size; pw=W//patch_size
    latents=latents.reshape(B,-1,pf,patch_size_t,ph,patch_size,pw,patch_size)
    latents=latents.permute(0,2,4,6,1,3,5,7).flatten(4,7).flatten(1,3)
    return latents

def unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
    B,S,D=latents.shape
    F=(num_frames-1)//8+1; H=height//32; W=width//32
    latents=latents.reshape(B,F,H,W,-1,patch_size_t,patch_size,patch_size)
    latents=latents.permute(0,4,1,5,2,6,3,7).flatten(6,7).flatten(4,5).flatten(2,3)
    return latents

# ═══ 完整生成 (手写三级调度) ═══
def generate_video(prompt, steps=50, frames=60, base_w=2560, base_h=1408,
                   fps=60, frame_rate=25, guidance_scale=3.0, seed=42, use_cache_embeds=True):
    t0=time.perf_counter()
    # ═══ T5 串行隔离: 加载→encode→卸载 (或直接用缓存 embeds 绕开 T5 段错误) ═══
    cache_path=r'D:\ai\_cached_embeds.pt'
    if use_cache_embeds and os.path.exists(cache_path):
        emb_data=torch.load(cache_path,weights_only=True)
        pos=emb_data['pos'].bfloat16(); neg=emb_data['neg'].bfloat16()
        pos_mask=emb_data['pos_mask'].bool(); neg_mask=emb_data['neg_mask'].bool()
        print(f'[gen] 用缓存 embeds (绕开T5)',flush=True)
    else:
        from transformers import T5EncoderModel, T5Tokenizer
        tok=T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer',local_files_only=True)
        te=T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder',torch_dtype=torch.bfloat16,
                                          local_files_only=True,low_cpu_mem_usage=True)
        enc=tok(prompt,padding='max_length',max_length=128,truncation=True,return_tensors='pt')
        encn=tok('',padding='max_length',max_length=128,truncation=True,return_tensors='pt')
        with torch.inference_mode():
            pos=te(input_ids=enc.input_ids,attention_mask=enc.attention_mask).last_hidden_state.bfloat16()
            neg=te(input_ids=encn.input_ids,attention_mask=encn.attention_mask).last_hidden_state.bfloat16()
        pos_mask=enc.attention_mask.bool(); neg_mask=encn.attention_mask.bool()
        del te, tok; gc.collect(); torch.cuda.empty_cache()
    emb_cat=torch.cat([neg,pos],dim=0).to(DEV)
    mask=torch.cat([neg_mask,pos_mask],dim=0).to(DEV)
    print(f'[gen] T5 encode done {time.perf_counter()-t0:.0f}s',flush=True)

    # ═══ latent 尺寸 ═══
    lat_F=(frames-1)//8+1; lat_H=base_h//32; lat_W=base_w//32
    seq_len=lat_F*lat_H*lat_W
    print(f'[gen] latent {lat_F}x{lat_H}x{lat_W} seq={seq_len}',flush=True)

    # ═══ scheduler ═══
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps
    sch=FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL,subfolder='scheduler')
    sigmas=np.linspace(1.0,1/steps,steps)
    mu=calculate_shift(seq_len,sch.config.get('base_image_seq_len',256),sch.config.get('max_image_seq_len',4096),
                       sch.config.get('base_shift',0.5),sch.config.get('max_shift',1.15))
    timesteps,_=retrieve_timesteps(sch,steps,'cpu',sigmas=sigmas,mu=mu)
    print(f'[gen] scheduler mu={mu:.3f} {time.perf_counter()-t0:.0f}s',flush=True)

    # ═══ 初始 latent (float32) ═══
    torch.manual_seed(seed)
    lat5=torch.randn(1,128,lat_F,lat_H,lat_W,dtype=torch.float32)
    latents=pack_latents(lat5).to(DEV)
    print(f'[gen] latent init {tuple(latents.shape)} {time.perf_counter()-t0:.0f}s',flush=True)

    # ═══ 去噪循环 (三级调度手写引擎) ═══
    with torch.inference_mode():
        for i,t in enumerate(timesteps):
            li=torch.cat([latents,latents],dim=0).bfloat16()
            ts=t.expand(2).to(DEV)
            rope=compute_rope(li,lat_F,lat_H,lat_W,8,32,frame_rate)
            np_=transformer_forward_sched(li,emb_cat,ts,mask,lat_F,lat_H,lat_W,rope_emb=rope,dtype=torch.bfloat16)
            u,c=np_.chunk(2)
            np_=u+guidance_scale*(c-u)
            latents=sch.step(np_.float(),t,latents.float(),return_dict=False)[0]
            if i%10==0 or i==steps-1:
                print(f'[gen] step {i+1}/{steps} t={t.item():.0f} std={latents.float().std().item():.3f} {time.perf_counter()-t0:.0f}s',flush=True)

    # ═══ 释放引擎资源 (去噪完成, VAE 前) ═══
    from ltx_sched import cleanup
    cleanup()
    print(f'[gen] engine cleanup done {time.perf_counter()-t0:.0f}s',flush=True)

    # ═══ VAE decode (pread加载 + tiling省显存) ═══
    from vae_loader import load_vae_pread
    vae=load_vae_pread(f'{MODEL}/vae_1024')  # pread加载, 在CPU
    lat5=unpack_latents(latents.cpu(),frames,base_h,base_w)
    m=vae.latents_mean.view(1,-1,1,1,1).to(lat5.dtype)
    s=vae.latents_std.view(1,-1,1,1,1).to(lat5.dtype)
    lat_denorm=lat5*s/vae.config.scaling_factor+m
    print(f'[gen] unpack {tuple(lat5.shape)} {time.perf_counter()-t0:.0f}s',flush=True)
    # tiling 必须在 .to(cuda) 之前启用: 分块decode, 避免大latent占满显存
    vae.enable_tiling()
    vae=vae.to(DEV)
    with torch.inference_mode():
        t_dec=torch.tensor([0.0],device=DEV,dtype=lat_denorm.dtype)
        # VAE 已对齐全量 f32 (值来自 bf16 checkpoint), decode 输入保持 f32
        decoded=vae.decode(lat_denorm.to(DEV),t_dec,return_dict=False)[0].cpu()
    print(f'[gen] decoded {tuple(decoded.shape)} {time.perf_counter()-t0:.0f}s',flush=True)
    fr=[decoded[0,:,i].permute(1,2,0).float() for i in range(decoded.shape[2])]
    fr=[(f+1)/2 for f in fr]
    from PIL import Image
    fr=[Image.fromarray((f.clamp(0,1)*255).to(torch.uint8).numpy()) for f in fr]

    # ═══ 保存 ═══
    fn=f'{OUT}/ltx_engine_{int(time.time())}.mp4'
    save_video_mp4(fr,fn,fps=fps)
    print(f'[gen] saved {fn} {fr[0].size} {len(fr)}f total {time.perf_counter()-t0:.0f}s',flush=True)
    return fn,fr[0].size,len(fr)

def save_video_mp4(frames,out_path,fps=60,crf=16):
    import imageio_ffmpeg,subprocess
    import numpy as np
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    w,h=frames[0].size
    cmd=[exe,'-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-',
         '-c:v','libx264','-crf',str(crf),'-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',out_path]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for f in frames:
        proc.stdin.write(np.array(f.convert('RGB'),dtype=np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    if proc.returncode!=0: raise RuntimeError(f'ffmpeg rc={proc.returncode}')

if __name__=='__main__':
    prompt=sys.argv[1] if len(sys.argv)>1 else 'a cute yellow orange cat sitting on a wooden table, bright colors, cinematic lighting, detailed fur'
    fn,size,n=generate_video(prompt)
    print(f'[gen] done -> {fn} {size} {n}f',flush=True)
