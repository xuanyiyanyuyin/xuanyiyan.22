# -*- coding: utf-8 -*-
"""验证: 纯文生图(无参考图)在当前引擎上能否跑通 — 用现成 embeds 缓存."""
import sys
sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2

fn = text2img_f2(
    prompt='a cute orange cat on white table, photorealistic',   # 与缓存对应
    steps=2, width=512, height=512,
    guidance_scale=4.0, resident_d=0,
    embeds_file=r'D:\ai\_cached_flux2_embeds_test.pt',
    seed=1,
)
print('RESULT:' + fn, flush=True)
