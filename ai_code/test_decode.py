import sys
sys.path.insert(0, r'D:\ai')
import flux_3tier as ft, torch
torch.cuda.empty_cache()
def mem(tag):
    a=torch.cuda.memory_allocated()/1024**3; r=torch.cuda.memory_reserved()/1024**3
    print(f'  [{tag}] alloc={a:.2f}GB reserved={r:.2f}GB', flush=True)
fn=ft.text2img_3tier('solo, single girl, cute cat, anime, simple', steps=6, width=384, height=384,
                     guidance_scale=7.0, resident_d=0, seed=1)
print('TEST_RESULT:'+fn, flush=True)
