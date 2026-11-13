# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_clean as M
import v22_pixel_512 as V

SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_512_complex_light", "pixel_512_best.pt")
src = V.load_src(SRC)
model = M.PixelEngine(dim=128).to(M.DEV)
model.load_state_dict(torch.load(CKPT, map_location=M.DEV, weights_only=True)); model.eval()

def stat(name, t):
    a = t.detach().cpu().numpy()
    print(f"  {name:14s} mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f} "
          f"p99={np.percentile(a,99):.4f} p1={np.percentile(a,1):.4f}")

for task in ["color", "denoise", "shift"]:
    inp, rule, tgt = V.make_pair(src, task, np.random.default_rng(43))
    with torch.no_grad():
        attr = model.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
        ps = model(inp.unsqueeze(0).to(M.DEV), attr, rule.unsqueeze(0).to(M.DEV))
        pred = V.apply_rule(ps, rule.unsqueeze(0).to(M.DEV))[0]
    tg = tgt.to(M.DEV)
    print(f"[{task}]")
    stat("raw_net(sigmoid)", ps[0])
    stat("pred(after rule)", pred)
    stat("target", tg)
    pl = M.srgb_to_linear(pred.unsqueeze(0))[0]; tl = M.srgb_to_linear(tg.unsqueeze(0))[0]
    stat("pred_linear", pl); stat("tgt_linear", tl)