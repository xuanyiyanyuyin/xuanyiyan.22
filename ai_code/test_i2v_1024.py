# 引擎级二分测试: 1024×1024 i2v (S=4096*2+512=8704)
import sys
sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2
REF = r'D:\ai\output\images\image_091881110277630.png'
EMBEDS = r'D:\ai\_embeds_miao_2k.pt'
PROMPT = 'a cute anime girl, silver fluffy hair, orange cat-ear headphones'
fn = text2img_f2(
    prompt=PROMPT, steps=8, width=1024, height=1024,
    guidance_scale=4.0, resident_d=0, images=[REF],
    embeds_file=EMBEDS, seed=1,
)
print('RESULT:' + fn, flush=True)
