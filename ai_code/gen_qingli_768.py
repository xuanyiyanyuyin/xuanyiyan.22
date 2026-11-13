import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
# 同 prompt, 原生分辨率768x1376, 常驻12, decode tiling开启
prompt = ("solo, single girl, centered full body standing portrait, vertical composition, "
          "Qing Li, long emerald green hair, soft pink eyes, gentle sweet smile, "
          "elegant light green kimono with wide sleeves and floral obi, holding a bouquet of red roses, "
          "standing under blooming cherry blossom tree, petals falling, warm spring sunshine, "
          "dreamy soft lighting, anime style, beautiful detailed eyes, masterpiece, 8k")
fn = ft.text2img_3tier(prompt, steps=50, width=768, height=1376, guidance_scale=7.0, resident_d=12)
print('T2I_RESULT:' + fn, flush=True)
