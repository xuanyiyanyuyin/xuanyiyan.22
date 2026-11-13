import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
DEV='cuda'
S=51712; H=48; HD=128
q=torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16)
k=torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16)
v=torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16)
for b in ['native','_native_flash','_native_efficient','_native_math','_native_cudnn','flex','flash']:
    torch.cuda.synchronize()
    try:
        out=dispatch_attention_fn(q,k,v,backend=b)
        torch.cuda.synchronize()
        print(f'{b}: OK out={tuple(out.shape)}', flush=True)
        del out
    except Exception as e:
        print(f'{b}: FAIL {type(e).__name__} {str(e)[:60]}', flush=True)
    torch.cuda.empty_cache()
print('BACKENDS_DONE', flush=True)
