import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
prompt = ("solo, single girl, centered full body standing portrait, vertical composition, "
          "Qing Li, long emerald green hair, soft pink eyes, gentle sweet smile, "
          "elegant light green kimono with wide sleeves and floral obi, holding a bouquet of red roses, "
          "standing under blooming cherry blossom tree, petals falling, warm spring sunshine, "
          "dreamy soft lighting, anime style, beautiful detailed eyes, masterpiece, 8k")
fn = ft.text2img_3tier(prompt, steps=50, width=3840, height=2160, guidance_scale=7.0, resident_d=1)
print('T2I_RESULT:' + fn, flush=True)
