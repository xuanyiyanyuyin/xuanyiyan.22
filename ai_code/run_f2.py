import sys, time
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
t0=time.time()
fn=f2.text2img_f2('a cute orange cat sitting on a wooden table, soft light', steps=6, width=512, height=512, resident_d=0)
print(f'F2_DONE in {time.time()-t0:.0f}s -> {fn}', flush=True)
