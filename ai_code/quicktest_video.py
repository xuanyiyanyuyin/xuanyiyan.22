"""
LTX 生视频最小测试 — 2B模型快速验证
用法: python quicktest_video.py "english prompt" [frames]
"""
import torch, sys, time, os, io
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HF_HUB_OFFLINE']='1'

MODEL=r'D:\ai\models\LTX-Video'
OUT=r'D:\ai\output\videos'
os.makedirs(OUT, exist_ok=True)

def main():
    prompt = sys.argv[1] if len(sys.argv)>1 else 'a red apple slowly rotating on a white table'
    frames = int(sys.argv[2]) if len(sys.argv)>2 else 17
    steps = int(sys.argv[3]) if len(sys.argv)>3 else 6
    print(f'[ltx] prompt: {prompt} frames={frames} steps={steps}', flush=True)
    t0 = time.perf_counter()

    from diffusers import LTXPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(f'{MODEL}/tokenizer', local_files_only=True)
    te = T5EncoderModel.from_pretrained(f'{MODEL}/text_encoder', torch_dtype=torch.bfloat16, local_files_only=True)
    vae = AutoencoderKLLTXVideo.from_pretrained(f'{MODEL}/vae', torch_dtype=torch.bfloat16, local_files_only=True)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder='scheduler')
    print(f'[ltx] components loaded {time.perf_counter()-t0:.0f}s', flush=True)

    # 13B主力权重 via single_file + ltx_config(48层13B配置)
    # low_cpu_mem_usage=True: 流式加载到meta, 避免28GB全量RAM
    tf = LTXVideoTransformer3DModel.from_single_file(
        f'{MODEL}/ltxv-13b-0.9.8-distilled.safetensors',
        torch_dtype=torch.bfloat16, config=r'D:\ai\ltx_config',
        low_cpu_mem_usage=True)
    print(f'[ltx] transformer(13B) loaded {time.perf_counter()-t0:.0f}s', flush=True)

    pipe = LTXPipeline(transformer=tf, scheduler=sch, vae=vae, text_encoder=te, tokenizer=tok)
    pipe.enable_sequential_cpu_offload(device='cuda')
    print(f'[ltx] offload ready, denoising...', flush=True)

    t1 = time.perf_counter()
    res = pipe(prompt=prompt, num_inference_steps=steps, num_frames=frames,
               height=512, width=512, guidance_scale=3.0,
               generator=torch.Generator('cpu').manual_seed(42))
    dt = time.perf_counter()-t1
    fr = res.frames[0]
    print(f'[ltx] denoised {len(fr)} frames in {dt:.0f}s', flush=True)

    fn = f'{OUT}/ltx_test_{int(time.time())}.gif'
    from PIL import Image
    imgs = [Image.fromarray(f) if not isinstance(f, Image.Image) else f for f in fr]
    imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=42, loop=0)
    print(f'[ltx] saved {fn} ({imgs[0].size}, {len(imgs)} frames)', flush=True)
    print(f'[ltx] total {time.perf_counter()-t0:.0f}s VRAM {torch.cuda.max_memory_allocated()/1024**3:.1f}GB', flush=True)

if __name__ == '__main__':
    main()
