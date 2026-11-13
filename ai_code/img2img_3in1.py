import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft
# 场景化: 3人樱花树下一起聊天喝茶, 高自由融合
prompt = ("three anime girls gathered around a low wooden table under a large cherry blossom tree, "
          "having tea and chatting happily, warm spring afternoon, pink petals falling, "
          "Xuan Yiyan with pink-gold hair and golden halo on the left, "
          "silver-blue haired girl with blue eyes in white kimono in the center holding a teacup, "
          "Qing Li with pastel green hair in sakura kimono on the right, "
          "unified harmonious scene, soft sunlight, detailed faces, anime style, masterpiece")
fn = ft.img2img_3tier(prompt, r'D:\ai\output\images\chara_3in1.png',
                      strength=0.95, steps=28, width=768, height=448, guidance_scale=7.0, resident_d=0)
print('I2I_RESULT:' + fn, flush=True)
