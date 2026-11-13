# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_v22full_complex6", "pixel_v22full_best.pt")
src = V.load_src(SRC)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
m = T.PixelEngineV22(dim=128).to(T.DEV)
ck = torch.load(CKPT, map_location=T.DEV, weights_only=True)
m.load_state_dict(ck["model"] if "model" in ck else ck); m.eval()
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
with torch.no_grad():
    attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
    pred_src, *_ = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
p = pred_src[0]; s = src_t[0]
lp = M.torch_rgb_to_lab(p.unsqueeze(0))[0]; ls = M.torch_rgb_to_lab(s.unsqueeze(0))[0]
print("=== pred_src vs 原图src_t 逐分量误差（全图均值）===")
print("linear RGB L1 (当前主loss) = %.4f" % F.l1_loss(M.srgb_to_linear(p), M.srgb_to_linear(s)).item())
print("sRGB RGB L1              = %.4f" % F.l1_loss(p, s).item())
print("Lab L  (明度) L1         = %.4f" % F.l1_loss(lp[0], ls[0]).item())
print("Lab a  (红绿色度) L1     = %.4f" % F.l1_loss(lp[1], ls[1]).item())
print("Lab b  (黄蓝色度) L1     = %.4f" % F.l1_loss(lp[2], ls[2]).item())
Cp = torch.sqrt(lp[1]**2+lp[2]**2+1e-6); Cs = torch.sqrt(ls[1]**2+ls[2]**2+1e-6)
print("色度幅值|C| L1           = %.4f" % F.l1_loss(Cp, Cs).item())
pn = p.permute(1,2,0).cpu().numpy(); sn = s.permute(1,2,0).cpu().numpy()
print("ΔE00 (感知色差)          = %.4f" % cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(sn)).mean())
print("目标平均色度|C|=%.2f  预测平均色度|C|=%.2f" % (Cs.mean().item(), Cp.mean().item()))