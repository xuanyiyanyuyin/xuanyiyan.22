# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_mdn as T
import v22_pixel_512 as V
m = T.PixelEngineMDN(dim=128, n_modes=5).to(T.DEV)
print("params", sum(p.numel() for p in m.parameters()))
src = V.load_src(None)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
attr = m.encode_attr([(src*255).astype(np.uint8)[:,:,::-1]])
means, log_w, log_s = m(src_t, attr, torch.zeros(1,16,device=T.DEV))
print("means", tuple(means.shape), "range", float(means.min()), float(means.max()))
loss = T.mdn_loss(means, log_w, log_s, src_t)
loss.backward()
pred = m.decode(means, log_w, log_s)
print("NLL", float(loss.detach()), "finite", bool(torch.isfinite(loss)), "decode", tuple(pred.shape), float(pred.min()), float(pred.max()))
print("VRAM_GB", round(torch.cuda.max_memory_allocated()/2**30, 2))