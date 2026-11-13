# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_clean as M
m = M.PixelEngine(dim=128).to(M.DEV)
print("params", sum(p.numel() for p in m.parameters()))
img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
attr = m.encode_attr([img])
print("attr", tuple(attr.shape), "finite", bool(torch.isfinite(attr).all()), "chroma_slots_nz", int((attr[0, 112+34:112+48].abs() > 1e-6).sum()))
x = torch.rand(1, 3, 512, 512, device=M.DEV)
r = torch.rand(1, 16, device=M.DEV)
p = m(x, attr, r)
loss = M.structural_loss(p, torch.rand_like(p))
loss.backward()
print("OK pred", tuple(p.shape), "range", float(p.min()), float(p.max()), "loss", float(loss), "finite", bool(torch.isfinite(loss)))
print("VRAM_GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))