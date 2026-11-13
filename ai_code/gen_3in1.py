import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
prompt = ("three girls standing side by side, Xuan Yiyan with long pink-gold hair and golden halo in pink kimono on the left, "
          "Miaoduo with long silver-blue hair and blue eyes in white modern kimono holding a pink rose in the middle, "
          "Qingli with long pastel green hair in sakura kimono with red rose on chest on the right, "
          "cherry blossom garden, spring sunshine, soft pastel colors, dreamy lighting, "
          "anime style, detailed eyes, masterpiece")
fn = ft.img2img_3tier(prompt, r'D:\ai\output\images\chara_3in1.png',
                      strength=0.5, steps=16, width=768, height=420, guidance_scale=7.0, resident_d=0)
print('I2I_RESULT:' + fn, flush=True)
