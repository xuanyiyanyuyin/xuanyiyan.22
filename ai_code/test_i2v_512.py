# -*- coding: utf-8 -*-
"""诊断: 512×512×8步 i2v, CUDA_LAUNCH_BLOCKING=1 精确定位 illegal memory access 的真起点."""
import sys
sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2

REF = r'D:\ai\output\images\image_091881110277630.png'
EMBEDS = r'D:\ai\_embeds_miao_2k.pt'
PROMPT = 'a cute anime girl, silver fluffy hair, orange cat-ear headphones'

fn = text2img_f2(
    prompt=PROMPT, steps=1, width=2560, height=2560,    # 2K ref-1024: 一步验证步时
    guidance_scale=4.0, resident_d=0, images=[REF],
    embeds_file=EMBEDS, seed=1,
)
print('RESULT:' + fn, flush=True)
