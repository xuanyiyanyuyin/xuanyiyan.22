from PIL import Image
import os
# 3个角色立绘 (用高清版)
imgs = [
    r'D:\ai\output\images\t2i_1788525031.png',   # 暄依嫣 900x1600 (竖版)
    r'D:\ai\output\images\t2i_1788727783.png',   # 喵朵 (你发的另一张? 512图)
    r'D:\ai\output\images\t2i_1788730008.png',   # 青璃 512x896
]
# 用你发的2张高清temp图替换喵朵/暄依嫣
# 但temp图不是固定路径, 先试现有; 若你要用temp, 告诉我路径
H=1400
pil=[]
for p in imgs:
    if os.path.exists(p):
        im=Image.open(p).convert('RGB'); pil.append((os.path.basename(p),im))
pil2=[(n, im.resize((int(im.width*H/im.height),H))) for n,im in pil]
total_w=sum(im.width for _,im in pil2)+40
canvas=Image.new('RGB',(total_w,H),(245,245,245))
x=0
for n,im in pil2:
    canvas.paste(im,(x,0)); x+=im.width+20
out=r'D:\ai\output\images\chara_3in1.png'
canvas.save(out)
print('拼接完成:', out, canvas.size)
for n,im in pil2: print(f'  {n} -> {im.size}')
