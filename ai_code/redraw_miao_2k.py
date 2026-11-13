# -*- coding: utf-8 -*-
"""FLUX.2 图生图重绘 2560×2560 (50步). 用已编码的 embeds 缓存, 不再加载 text_encoder.
注意: 编码(_embeds_miao_2k.pt)已单独跑过; 此脚本只读 embeds + 加载 transformer, 避免 text_encoder/transformer 同场 OOM."""
import sys
sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2

REF = r'D:\ai\output\images\image_091881110277630.png'
EMBEDS = r'D:\ai\_embeds_miao_2k.pt'     # 已编码好的喵朵 embeds(单独跑过)
PROMPT = ('a cute anime girl, silver fluffy hair, orange cat-ear headphones, '
          'pink bow, blue maid dress, blue collar, golden eyes, '
          'soft lighting, clean background, high resolution, sharp details, '
          'masterpiece, portrait, centered')

fn = text2img_f2(
    prompt=PROMPT, steps=50, width=2560, height=2560,
    guidance_scale=4.0, resident_d=0, images=[REF],
    embeds_file=EMBEDS, seed=1,          # 读已编码 embeds, 不重编码
)
print('RESULT:' + fn, flush=True)
