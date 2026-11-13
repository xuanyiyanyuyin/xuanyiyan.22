"""
FLUX.1-dev 魔改生图 — 12GB显存流式版
策略: diffusers官方计算逻辑 + 逐模块CPU卸载(GPU只放当前块, 算完搬回)
用法: D:\ai\python\python.exe gen_image.py "一只在月亮上打太极的橘猫" [steps] [size]
"""
import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
# ══ [2026-10-02] 内存卫生：mmap 进来的权重页是干净的，用完/退出时归还给系统 ══
#    自带 sys.path 兜底 —— 放哪都不会 ImportError（2026-10-02 踩过：块被插到
#    sys.path.insert 之前，导致 flux2_encode_gen 一加载就 ModuleNotFoundError）
try:
    import sys as _mh_sys
    if r'D:\ai' not in _mh_sys.path:
        _mh_sys.path.insert(0, r'D:\ai')
    import mem_hygiene as _mh
    _mh.trim('start')
    import atexit as _atexit
    _atexit.register(lambda: _mh.trim('exit'))
except Exception:
    _mh = None

MODEL=r'D:\ai\models\FLUX.1-dev'
OUT=r'D:\ai\output\images'
os.makedirs(OUT, exist_ok=True)
# ═══ IP-Adapter 配置(在 get_pipe 前定义, 避免 NameError) ═══
IPADAPTER_PATH=r'D:\ai\models\adapter\flux-ip-adapter'
CLIP_VIT_PATH=r'D:\ai\models\adapter\clip-vit-large-patch14'
_IPCACHE={'ipadapter_loaded':False}

def _allow_long_seq(pipe):
    """放宽 check_inputs 的 max_sequence_length>512 硬限, 支持长上下文4096.
    diffusers 0.40 保守卡512, 但 FLUX T5 tokenizer 实际可处理更长."""
    try:
        orig = pipe.check_inputs
        def patched(prompt, prompt_2=None, negative_prompt=None, negative_prompt_2=None,
                    prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None, callback_on_step_end_tensor_inputs=None,
                    max_sequence_length=None):
            # 同样校验, 只是把 max_sequence_length>512 的硬限放宽到 4096
            if max_sequence_length is not None and max_sequence_length > 4096:
                raise ValueError(f"max_sequence_length cannot be greater than 4096 but is {max_sequence_length}")
            return True
        pipe.check_inputs = patched
    except Exception as e:
        print(f'[gen] warn: allow_long_seq failed: {e}', flush=True)
    return pipe

def main():
    prompt=sys.argv[1] if len(sys.argv)>1 else '一只在月亮上打太极的橘猫, 电影级光影'
    steps=int(sys.argv[2]) if len(sys.argv)>2 else 50
    iw=int(sys.argv[3]) if len(sys.argv)>3 else 2560
    ih=int(sys.argv[4]) if len(sys.argv)>4 else 1408
    print(f'[gen] prompt: {prompt}',flush=True)
    print(f'[gen] steps={steps} size={iw}x{ih}',flush=True)

    from diffusers import FluxPipeline
    t0=time.perf_counter()
    pipe=FluxPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True)
    print(f'[gen] pipeline loaded {time.perf_counter()-t0:.0f}s, 启用逐块显存搬运...',flush=True)
    _allow_long_seq(pipe)
    # 魔改核心: 每个子模块(每个transformer块/T5块)用时上GPU, 用完回CPU
    # GPU峰值 = 单个块(~400MB) + 激活, 12GB绰绰有余
    pipe.enable_sequential_cpu_offload(device='cuda')
    print(f'[gen] offload ready, 开始去噪...',flush=True)

    t1=time.perf_counter()
    res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
             height=ih,width=iw,max_sequence_length=4096,
             generator=torch.Generator('cpu').manual_seed(42))
    dt=time.perf_counter()-t1
    img=res.images[0]
    fn=f'{OUT}/flux_{int(time.time())}.png'
    img.save(fn)
    print(f'[gen] done in {dt:.0f}s ({dt/steps:.1f}s/step) -> {fn}',flush=True)
    print(f'[gen] VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB',flush=True)

_pipe=None
def get_pipe(ipadapter=False):
    """全局单例FLUX pipeline, 避免重复加载.
    ipadapter=True 时: from_pretrained → load_ip_adapter → enable_offload(顺序对, 不meta)"""
    global _pipe
    if _pipe is not None:
        return _pipe
    import os
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from diffusers import FluxPipeline
    torch=__import__('torch')
    _pipe=FluxPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True)
    if ipadapter:
        # 先加载 IP-Adapter (此时 transformer 是真实权重, 可正确注入 ip_attention)
        _pipe.load_ip_adapter(
            IPADAPTER_PATH, weight_name='ip_adapter.safetensors',
            image_encoder_pretrained_model_name_or_path=CLIP_VIT_PATH,
            image_encoder_dtype=torch.bfloat16,
            low_cpu_mem_usage=False, local_files_only=True,
        )
        try:
            _pipe.image_encoder.to('cuda'); _pipe.image_encoder.eval()
        except Exception:
            pass
        try:
            _pipe.set_ip_adapter_scale(0.8)
        except Exception:
            pass
        _IPCACHE['ipadapter_loaded']=True
    _allow_long_seq(_pipe)
    _pipe.enable_sequential_cpu_offload(device='cuda')
    return _pipe

def free_pipe():
    """释放FLUX引擎, 归还显存给122B"""
    global _pipe
    if _pipe is not None:
        del _pipe; _pipe=None
        import torch; torch.cuda.empty_cache()
        print('[gen] FLUX released',flush=True)

# ═══ IP-Adapter 加装(不新建引擎, 复用 _pipe 单例) ═══
def add_ipadapter():
    """加装 IP-Adapter: 若_pipe未加载或已offload, 用 get_pipe(ipadapter=True) 重建正确顺序.
    已加载则直接返回(引擎默认已带 adapter)。"""
    global _IPCACHE
    if _IPCACHE['ipadapter_loaded'] and _pipe is not None:
        return True
    # 若 _pipe 已存在但未带 adapter, 释放重建为 ipadapter=True
    if _pipe is not None:
        free_pipe()
    get_pipe(ipadapter=True)
    return True

def _encode_ip_reference(img_pil):
    """算参考图 image_embeds。image_encoder 由 add_ipadapter 加载进 _pipe(常驻GPU)。
    手动预处理(不依赖 CLIPImageProcessor.from_pretrained, 本地缺preprocessor_config)。
    照你的模式: 只编码这一次, 用完把image_encoder挪回offload/释放, 不常驻抢显存。"""
    torch=__import__('torch')
    import torch.nn.functional as F
    add_ipadapter()                      # 确保 IP-Adapter 就绪(必要时重建为 ipadapter=True)
    pipe=get_pipe(ipadapter=True)
    # image_encoder 在 GPU (ipadapter=True 时加载), 手动预处理
    img=img_pil.convert('RGB')
    import torchvision.transforms as T
    tfs=T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                    std=[0.26862954,0.26130258,0.27577711]),
    ])
    x=tfs(img).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
    if hasattr(pipe,'image_encoder') and pipe.image_encoder is not None:
        enc=pipe.image_encoder
    else:
        return None
    enc.eval()
    with torch.no_grad():
        embeds=enc(pixel_values=x).image_embeds
    embeds=embeds.unsqueeze(1).to('cpu')   # (1,1,dim) 挪回CPU, 不占显存
    return embeds

def generate_image(prompt, steps=50, out_path=None, size=2560, height=1408, ip_reference=None):
    """可导入的生图函数(供 server 调用). ip_reference=PIL图 时启用 IP-Adapter 角色一致"""
    from PIL import Image
    t0=time.perf_counter()
    pipe=get_pipe()
    kwargs={}
    if ip_reference is not None:
        add_ipadapter()                      # 加装 IP-Adapter(复用_pipe)
        embeds=_encode_ip_reference(ip_reference)   # 算embeds(用完即释放image_encoder)
        kwargs['ip_adapter_image_embeds']=embeds
    res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
             height=height,width=size,max_sequence_length=4096,
             generator=torch.Generator('cpu').manual_seed(int(time.time())%10000),
             **kwargs)
    img=res.images[0]
    if out_path is None:
        fn=f'{OUT}/chat_{int(time.time())}.png'
    else:
        fn=out_path
    img.save(fn)
    print(f'[gen] {fn} ({img.size}) {time.perf_counter()-t0:.0f}s',flush=True)
    return fn

def generate_batch(prompt, steps=50, count=4, size=2560, height=1408):
    """批量生成 count 张图(不同seed), 返回路径列表"""
    t0=time.perf_counter()
    pipe=get_pipe()
    base=int(time.time())%100000
    paths=[]
    for i in range(count):
        fn=f'{OUT}/gen_{base}_{i}.png'
        res=pipe(prompt=prompt,num_inference_steps=steps,guidance_scale=3.5,
                 height=height,width=size,max_sequence_length=4096,
                 generator=torch.Generator('cpu').manual_seed(base+i))
        res.images[0].save(fn)
        paths.append(fn)
        print(f'[gen] {i+1}/{count} {fn} ({time.perf_counter()-t0:.0f}s)',flush=True)
    return paths

# ═══ 图生图 / 高清修复 (FluxImg2ImgPipeline 独立单例, 不含IP-Adapter那2.5G, 12G安全) ═══
_img2pipe=None
def get_img2pipe():
    """图生图 pipeline (独立于文生图 _pipe). 顺序卸载, 不加载IP-Adapter image_encoder, 不会OOM."""
    global _img2pipe
    if _img2pipe is not None:
        return _img2pipe
    import os
    os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    from diffusers import FluxImg2ImgPipeline
    torch=__import__('torch')
    _img2pipe=FluxImg2ImgPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True)
    _img2pipe.enable_sequential_cpu_offload(device='cuda')
    print('[gen] img2img pipeline ready',flush=True)
    return _img2pipe

def free_img2pipe():
    """释放图生图引擎, 归还显存"""
    global _img2pipe
    if _img2pipe is not None:
        del _img2pipe; _img2pipe=None
        import torch; torch.cuda.empty_cache()
        print('[gen] img2img released',flush=True)

def img2img_refine(prompt, init_image, strength=0.35, steps=28, width=768, height=1376,
                   guidance_scale=3.5, out_path=None):
    """FLUX 图生图/重绘修复. init_image=PIL或路径.
      strength低(0.30-0.40)=修细节保结构; 高=大改. 用于高清修复角色图/修复放大糊面."""
    from PIL import Image
    import torch
    if isinstance(init_image, str):
        init_image=Image.open(init_image).convert('RGB')
    pipe=get_img2pipe()
    t0=time.perf_counter()
    res=pipe(prompt=prompt, image=init_image, strength=strength,
             num_inference_steps=steps, guidance_scale=guidance_scale,
             height=height, width=width,
             generator=torch.Generator('cpu').manual_seed(int(time.time())%10000))
    img=res.images[0]
    fn=out_path or f'{OUT}/refine_{int(time.time())}.png'
    img.save(fn)
    print(f'[gen] img2img {fn} ({img.size}) strength={strength} steps={steps} {time.perf_counter()-t0:.0f}s',flush=True)
    return fn

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--i2i':
        # python gen_image.py --i2i <image> "<prompt>" [strength] [steps] [out]
        img=sys.argv[2]
        prompt=sys.argv[3] if len(sys.argv)>3 else '高清重绘, 保持人物外观一致, 精致细节'
        strength=float(sys.argv[4]) if len(sys.argv)>4 else 0.35
        steps=int(sys.argv[5]) if len(sys.argv)>5 else 28
        out=sys.argv[6] if len(sys.argv)>6 else None
        print(f'[img2img] {img} strength={strength} steps={steps} width=768 height=1376',flush=True)
        fn=img2img_refine(prompt, img, strength=strength, steps=steps, width=768, height=1376, out_path=out)
        print('[IMG2IMG_RESULT]'+fn,flush=True)
    else:
        main()
