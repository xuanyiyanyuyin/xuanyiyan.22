# -*- coding: utf-8 -*-
"""短探针：从E400续训+400轮，每100轮量color CE，测衰减率供外推。"""
import os, sys, time, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T, v22_pixel_clean as M, v22_pixel_512 as V, color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_v22full_complex16", "pixel_v22full_best.pt")
src = V.load_src(SRC); src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
m = T.PixelEngineV22(dim=128, n_colors=512).to(T.DEV)
ck = torch.load(CKPT, map_location=T.DEV, weights_only=True)
m.load_state_dict(ck["model"]); m.train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
rng = np.random.default_rng(42)
tasks = ["color","shift","flip","combo","denoise"]
cinp, crule, ctgt = V.make_pair(src, "color", np.random.default_rng(43))
cinp = cinp.unsqueeze(0).to(T.DEV); crule = crule.unsqueeze(0).to(T.DEV)
def color_ce():
    m.eval()
    with torch.no_grad():
        attr = m.encode_attr([(cinp[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
        pred, logits, *_ = m(cinp, attr, crule)
        sf = src_t.permute(0,2,3,1).reshape(-1,3); cb = m.color_codebook
        d = (sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t()
        ti = d.argmin(1).reshape(1,512,512)
        ce = F.cross_entropy(logits, ti).item()
        pn = pred[0].permute(1,2,0).cpu().numpy()
        de = cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(src)).mean()
    m.train(); return ce, de
ce0, de0 = color_ce(); print(f"E400(起点): colorCE={ce0:.3f} ΔE={de0:.2f}", flush=True)
t0=time.time()
for ep in range(1, 401):
    inp, rule, tgt = V.make_pair(src, tasks[(ep-1)%5], rng)
    inp=inp.unsqueeze(0).to(T.DEV); rule=rule.unsqueeze(0).to(T.DEV)
    attr = m.encode_attr([(inp[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
    pred, logits, ct, ac, st, gate = m(inp, attr, rule)
    with torch.no_grad():
        sf=src_t.permute(0,2,3,1).reshape(-1,3); cb=m.color_codebook
        d=(sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t(); ti=d.argmin(1).reshape(1,512,512)
    loss=F.cross_entropy(logits,ti)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if ep%100==0:
        ce,de=color_ce(); print(f"E{400+ep}: colorCE={ce:.3f} ΔE={de:.2f} {time.time()-t0:.0f}s", flush=True)