# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_v22full_complex5", "pixel_v22full_best.pt")
src = V.load_src(SRC)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
m = T.PixelEngineV22(dim=128).to(T.DEV)
ck = torch.load(CKPT, map_location=T.DEV, weights_only=True)
m.load_state_dict(ck["model"] if "model" in ck else ck); m.eval()
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
with torch.no_grad():
    attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
    pred_src, ct, ac, st, gate = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
    pred_exam = V.apply_rule(pred_src, rule.unsqueeze(0).to(T.DEV))
# 番茄区：目标Lab a>20 的像素
lt = cs.rgb_to_lab(src)
red = (lt[...,1]>20)&(lt[...,0]>30)
ps = pred_src[0].permute(1,2,0).cpu().numpy()
pe = pred_exam[0].permute(1,2,0).cpu().numpy()
print("原图src_t 番茄区RGB均值 =", src[red].mean(0).round(3), "(应偏红)")
print("网络pred_src 番茄区RGB均值 =", ps[red].mean(0).round(3))
print("remap后pred_exam 番茄区RGB均值 =", pe[red].mean(0).round(3))
print("训练主loss L1(linear pred_src, linear src_t) = %.4f" % F.l1_loss(M.srgb_to_linear(pred_src), M.srgb_to_linear(src_t)).item())
print("pred_src vs src_t 全图ΔE = %.2f" % cs.delta_e_2000(cs.rgb_to_lab(ps), lt).mean())
print("pred_exam vs tgt(remap src) 全图ΔE = %.2f" % cs.delta_e_2000(cs.rgb_to_lab(pe), cs.rgb_to_lab(tgt.to(T.DEV).permute(1,2,0).cpu().numpy())).mean())
print("rule mult =", rule[1:4].numpy().round(2), "color激活=", float(rule[0]))