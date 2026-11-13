# -*- coding: utf-8 -*-
"""纯计算收敛判据：逐像素 CE loss ↔ ΔE 对应，算出 ΔE<2 需要的 CE 水平。不训练。"""
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T, v22_pixel_clean as M, v22_pixel_512 as V, color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_v22full_complex16", "pixel_v22full_best.pt")
src = V.load_src(SRC); src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
m = T.PixelEngineV22(dim=128, n_colors=512).to(T.DEV)
ck = torch.load(CKPT, map_location=T.DEV, weights_only=True)
m.load_state_dict(ck["model"] if "model" in ck else ck); m.eval()
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
with torch.no_grad():
    attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
    pred, logits, *_ = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
    sf = src_t.permute(0,2,3,1).reshape(-1,3); cb = m.color_codebook
    d = (sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t()
    ti = d.argmin(1).reshape(1,512,512)
    ce = F.cross_entropy(logits, ti, reduction='none')[0]      # [H,W] 逐像素CE
pn = pred[0].permute(1,2,0).cpu().numpy()
de = cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(src))   # [H,W] 逐像素ΔE
ce_np = ce.cpu().numpy()
print("当前 color: 平均CE=%.3f 平均ΔE=%.2f ΔE<2占比=%.1f%%" % (ce_np.mean(), de.mean(), (de<2).mean()*100))
print("逐像素 CE分箱 → 平均ΔE / ΔE<2占比:")
for lo, hi in [(0,0.1),(0.1,0.3),(0.3,0.5),(0.5,1.0),(1.0,2.0),(2.0,4.0),(4.0,99)]:
    mask = (ce_np>=lo)&(ce_np<hi)
    if mask.sum()>0:
        print(f"  CE[{lo:.1f},{hi:.1f}): 像素{mask.sum():6d} 平均ΔE={de[mask].mean():6.2f} ΔE<2占{(de[mask]<2).mean()*100:5.1f}%")
# 收敛判据：ΔE<2 的像素其CE分布
good = de<2
print("ΔE<2像素的CE: 均值=%.3f 中位=%.3f p90=%.3f" % (ce_np[good].mean(), np.median(ce_np[good]), np.percentile(ce_np[good],90)))