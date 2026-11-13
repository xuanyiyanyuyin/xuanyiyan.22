import sys, os
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
fn = ft.img2img_3tier('高清重绘, 保持人物一致, 精致细节', r'D:\ai\frontend\assets\qingli.png',
                      strength=0.18, steps=50, width=2560, height=1408, guidance_scale=7.0, resident_d=0)
print('RESULT:' + fn, flush=True)
