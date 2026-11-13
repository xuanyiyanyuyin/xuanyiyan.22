# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_mdn as T
import v22_pixel_512 as V
import color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_mdn_complex2", "pixel_mdn_best.pt")
src = V.load_src(SRC)
m = T.PixelEngineMDN(dim=128, n_modes=5).to(T.DEV)
m.load_state_dict(torch.load(CKPT, map_location=T.DEV, weights_only=True)); m.eval()
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
with torch.no_grad():
    attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
    means, log_w, log_s = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
sig = log_s.exp().clamp(0.02, 1.0)
pi = F.softmax(log_w, dim=1)
ent = -(pi * (pi+1e-9).log()).sum(1).mean()   # 权重熵（ln5=1.61=均匀）
print("sigma: mean=%.4f min=%.4f max=%.4f | 贴下限0.02占比=%.1f%%" % (sig.mean(), sig.min(), sig.max(), (sig<=0.021).float().mean()*100))
print("模式权重熵=%.3f (ln5=1.609=完全均匀)" % ent)
print("means范围: min=%.3f max=%.3f std=%.3f" % (means.min(), means.max(), means.std()))
pred = m.decode(means, log_w, log_s)[0]
p = pred.permute(1,2,0).cpu().numpy(); t = tgt.to(T.DEV).permute(1,2,0).cpu().numpy()
lp = cs.rgb_to_lab(p); lt = cs.rgb_to_lab(t)
red = (lt[...,1]>20)&(lt[...,0]>30)
print("目标红像素 目标Lab=%s 预测Lab=%s" % (lt[red].mean(0).round(1), lp[red].mean(0).round(1)))