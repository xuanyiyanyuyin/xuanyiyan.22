# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_token as T
import v22_pixel_512 as V
m = T.PixelTokenEngine(dim=128, n_bins=64).to(T.DEV)
print("params", sum(p.numel() for p in m.parameters()))
src = V.load_src(None)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
src_bins = T.to_bins(src_t, 64)
img = (src*255).astype(np.uint8)
attr = m.encode_attr([img[:,:,::-1]])
x = src_t
logits = m(x, attr, torch.zeros(1,16,device=T.DEV))
print("logits", tuple(logits.shape))
loss = F.cross_entropy(logits.reshape(-1,64), src_bins.reshape(-1))
loss.backward()
pred = m.decode(logits)
print("CE", float(loss), "finite", bool(torch.isfinite(loss)), "decode_range", float(pred.min()), float(pred.max()))
print("VRAM_GB", round(torch.cuda.max_memory_allocated()/2**30, 2))