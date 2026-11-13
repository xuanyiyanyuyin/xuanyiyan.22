# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T, v22_pixel_clean as M, v22_pixel_512 as V, color_science as cs
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_converge", "model.pt")
OUT = os.path.join(HERE, "pixel_converge", "render"); os.makedirs(OUT, exist_ok=True)
src = V.load_src(SRC)
m = T.PixelEngineV22(dim=128, n_colors=512).to(T.DEV)
ck = torch.load(CKPT, map_location=T.DEV, weights_only=True)
m.load_state_dict(ck["model"] if "model" in ck else ck); m.eval()
for task in ["color","shift","flip","denoise"]:
    inp, rule, tgt = V.make_pair(src, task, np.random.default_rng(43))
    with torch.no_grad():
        attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
        pred, logits, *_ = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
        pred = V.apply_rule(pred, rule.unsqueeze(0).to(T.DEV))[0]
    tg = tgt.to(T.DEV)
    p_np = pred.permute(1,2,0).cpu().numpy(); t_np = tg.permute(1,2,0).cpu().numpy()
    de = cs.delta_e_2000(cs.rgb_to_lab(p_np), cs.rgb_to_lab(t_np))
    trip = torch.cat([inp, pred.cpu(), tg.cpu()], dim=2).permute(1,2,0).numpy()
    big = (cs.floyd_steinberg(np.clip(trip,0,1))*255).round().astype(np.uint8)
    cs.save_png_with_icc(os.path.join(OUT, f"{task}.png"), big)
    print(f"{task}: ΔE={de.mean():.2f} 对齐{(de<2).mean()*100:.0f}%")
print("saved to", OUT)