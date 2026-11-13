import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
prompt = ("solo, single girl, full body, vertical, Qingli, very long luminous pastel green hair, "
          "hair falling completely behind her back in one continuous wave, no buns, no pigtails, "
          "hair loose and unbound, side-swept bangs, mature yamato nadeshiko style, elegant sakura kimono, "
          "red rose on the chest, cherry blossoms, floating petals, soft pastel colors, "
          "dreamy soft lighting, semi-realistic anime style, detailed eyes, masterpiece")
fn = ft.text2img_3tier(prompt, steps=20, width=512, height=880, guidance_scale=7.0, resident_d=0)
print('T2I_RESULT:' + fn, flush=True)
