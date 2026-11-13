# 二分测试: 2048×2048 i2v (S=128*128*2+512=33792)
import sys
sys.path.insert(0, r'D:\ai')
from flux2_3tier import text2img_f2
REF = r'D:\ai\output\images\image_091881110277630.png'
EMBEDS = r'D:\ai\_embeds_miao_2k.pt'
PROMPT = 'a cute anime girl, silver fluffy hair, orange cat-ear headphones'
fn = text2img_f2(
    prompt=PROMPT, steps=4, width=2048, height=2048,
    guidance_scale=4.0, resident_d=0, images=[REF],
    embeds_file=EMBEDS, seed=1,
)
print('RESULT:' + fn, flush=True)
