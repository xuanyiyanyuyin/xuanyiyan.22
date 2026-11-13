# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
m = T.PixelEngineV22(dim=128, n_colors=256).to(T.DEV)
print("params", sum(p.numel() for p in m.parameters()))
src = V.load_src(None)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
attr = m.encode_attr([(src*255).astype(np.uint8)[:,:,::-1]])
pred, logits, ct, ac, st, gate = m(src_t, attr, torch.zeros(1,16,device=T.DEV))
print("pred", tuple(pred.shape), "logits", tuple(logits.shape))
with torch.no_grad():
    sf = src_t.permute(0,2,3,1).reshape(-1,3); cb = m.color_codebook
    dist = (sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t()
    tidx = dist.argmin(1).reshape(1,src_t.shape[2],src_t.shape[3])
loss = F.cross_entropy(logits, tidx)
loss.backward()
hard = m.decode_hard(logits)
print("CE", float(loss.detach()), "hard", tuple(hard.shape), float(hard.min()), float(hard.max()), "finite", bool(torch.isfinite(loss)))
print("VRAM_GB", round(torch.cuda.max_memory_allocated()/2**30,2))