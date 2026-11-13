import torch
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed, apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
DEV='cuda'
S=51712; H=48; HD=128
pos=Flux2PosEmbed(theta=10000, axes_dim=[32,32,32,32]).to(DEV)
ids=torch.zeros(S,4,dtype=torch.int64,device=DEV)
idx=torch.arange(S,device=DEV); ids[:,1]=idx%160; ids[:,2]=(idx//160)%160
cos,sin=pos(ids)
q=torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16)
q=apply_rotary_emb(q,(cos,sin),sequence_dim=1)
print(f'rope后 is_contiguous={q.is_contiguous()} stride={tuple(q.stride())}', flush=True)
k=apply_rotary_emb(torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16),(cos,sin),sequence_dim=1)
v=torch.randn(1,H,S,HD,device=DEV,dtype=torch.bfloat16)
for b in ['native','_native_efficient','_native_math','_native_cudnn','flex']:
    torch.cuda.synchronize()
    try:
        out=dispatch_attention_fn(q,k,v,backend=b)
        torch.cuda.synchronize()
        print(f'noncontig {b}: OK', flush=True)
        del out
    except Exception as e:
        print(f'noncontig {b}: FAIL {type(e).__name__} {str(e)[:50]}', flush=True)
    torch.cuda.empty_cache()
print('STRIDE_DONE', flush=True)
