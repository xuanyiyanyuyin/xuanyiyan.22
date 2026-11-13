import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
prompt = "solo, single girl, Qing Li, long emerald hair, pink eyes, light green kimono, red roses, cherry blossoms, anime, detailed eyes"
fn = ft.text2img_3tier(prompt, steps=8, width=512, height=512, guidance_scale=7.0, resident_d=0)
print('T2I_RESULT:' + fn, flush=True)
