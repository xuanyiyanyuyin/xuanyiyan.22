import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
# 关键构图指令前置(CLIP只读前77 token), 单人+立绘放最前
prompt = ("solo, single girl, centered full body standing portrait, vertical, "
          "Xuan Yiyan, long pink-gold hair, twin side locks, golden halo, pink eyes, "
          "gentle smile, pink kimono, wide sleeves, floral hair ornament, cherry blossoms, "
          "spring sunshine, dreamy light, anime, detailed eyes")
fn = ft.text2img_3tier(prompt, steps=50, width=1408, height=2560, guidance_scale=7.0, resident_d=0)
print('T2I_RESULT:' + fn, flush=True)
