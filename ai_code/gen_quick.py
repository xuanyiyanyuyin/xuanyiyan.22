import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
# 512x512 + 8步, 快速测解码/网格
prompt = ("solo, single girl, Qing Li, long emerald green hair, pink eyes, gentle smile, "
          "light green kimono, red roses, cherry blossoms, anime style, detailed eyes, 8k")
fn = ft.text2img_3tier(prompt, steps=8, width=512, height=512, guidance_scale=7.0, resident_d=12)
print('T2I_RESULT:' + fn, flush=True)
