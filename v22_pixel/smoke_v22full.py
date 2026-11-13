# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
m = T.PixelEngineV22(dim=128).to(T.DEV)
master = T.MasterGate().to(T.DEV)
print("params", sum(p.numel() for p in m.parameters()), "master", sum(p.numel() for p in master.parameters()))
src = V.load_src(None)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
src_attr = torch.tensor(np.nan_to_num(M.P3Vis(chroma=True).process_image((src[:,:,::-1]*255).astype(np.uint8)),nan=0.,posinf=0.,neginf=0.),dtype=torch.float32).unsqueeze(0).to(T.DEV)
mask = (src_attr.abs().sum(0)>1e-6).float()
attr = m.encode_attr([(src*255).astype(np.uint8)[:,:,::-1]])
pred, ct, ac, st, gate = m(src_t, attr, torch.zeros(1,16,device=T.DEV))
print("pred", tuple(pred.shape), "ct", tuple(ct.shape), "ac", tuple(ac.shape), "st", tuple(st.shape))
main = F.l1_loss(M.srgb_to_linear(pred), M.srgb_to_linear(src_t))
lB = F.mse_loss(ct*mask, src_attr*mask).clamp(max=10)
lC = F.mse_loss(ac, src_attr[:,:48]).clamp(max=10)
lA = F.mse_loss(st, src_attr[:,48:112]).clamp(max=10)
gl = (gate.std()*10).clamp(0.01,1.0)
temps = master(torch.stack([gl.detach(),lA.detach(),lB.detach(),lC.detach(),main.detach()]))
loss = main + (temps[1]*lA+temps[2]*lB+temps[3]*lC) + temps[0]*gl
loss.backward()
print("loss", float(loss.detach()), "temps", temps.detach().cpu().numpy().round(2).tolist(), "finite", bool(torch.isfinite(loss)))
print("VRAM_GB", round(torch.cuda.max_memory_allocated()/2**30,2))