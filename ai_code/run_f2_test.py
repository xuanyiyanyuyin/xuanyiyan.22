import sys
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
fn=f2.text2img_f2('a cute orange cat on white table, photorealistic', steps=8, width=512, height=512, resident_d=0)
print('F2_RESULT:'+fn, flush=True)
