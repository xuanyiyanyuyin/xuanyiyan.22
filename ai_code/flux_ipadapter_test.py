# -*- coding: utf-8 -*-
"""FLUX IP-Adapter: 用 device_map='auto' 自动分配, 解决 image_encoder 的 CUDA/CPU 调度"""
import torch, time, os
from diffusers import FluxPipeline
from PIL import Image

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL = r'D:\ai\models\FLUX.1-dev'
IPADAPTER = r'D:\ai\models\adapter\flux-ip-adapter'
REF = r'D:\ai\output\角色库\miao_1788351147.png'

t0 = time.perf_counter()
print('[1] 加载 FLUX.1-dev (model offload)...', flush=True)
pipe = FluxPipeline.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
pipe.enable_model_cpu_offload()   # image_encoder 会真实在 CPU(可 .to 到 cuda)
print(f'[2] 基础管线 OK {time.perf_counter()-t0:.0f}s', flush=True)

print('[3] 加载 IP-Adapter...', flush=True)
pipe.load_ip_adapter(
    IPADAPTER,
    weight_name='ip_adapter.safetensors',
    image_encoder_pretrained_model_name_or_path=r'D:\ai\models\adapter\clip-vit-large-patch14',
    image_encoder_dtype=torch.bfloat16,
)
# 生成前把 image_encoder 移到 GPU(它真实权重在 CPU, 可 .to)
pipe.image_encoder.to('cuda')
print(f'[4] IP-Adapter OK + image_encoder 移到GPU {time.perf_counter()-t0:.0f}s', flush=True)

# 参考图
ref = Image.open(REF).convert('RGB').resize((512, 512))
prompt = ("同一个银发青和服少女, 站在樱花树下回眸, 手持红玫瑰, 青色和服裙, "
          "精致二次元插画, 柔和高光, 高质量")
print('[5] 生成 IP-Adapter 结果(512x512, 20步)...', flush=True)
res = pipe(
    prompt=prompt,
    ip_adapter_image=ref,
    num_inference_steps=20,
    guidance_scale=3.5,
    true_cfg_scale=3.5,
    height=512, width=512,
    max_sequence_length=4096,
    generator=torch.Generator('cpu').manual_seed(777),
)
out = r'D:\ai\output\images\ipadapter_test.png'
res.images[0].save(out)
print(f'[6] 完成 {out} in {time.perf_counter()-t0:.0f}s', flush=True)
