# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs

SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
CKPT = os.path.join(HERE, "pixel_512_complex_arch2", "pixel_512_best.pt")
src = V.load_src(SRC)
model = M.PixelEngine(dim=128).to(M.DEV)
model.load_state_dict(torch.load(CKPT, map_location=M.DEV, weights_only=True)); model.eval()

for task in ["color", "shift"]:
    inp, rule, tgt = V.make_pair(src, task, np.random.default_rng(43))
    with torch.no_grad():
        attr = model.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
        ps = model(inp.unsqueeze(0).to(M.DEV), attr, rule.unsqueeze(0).to(M.DEV))
        pred = V.apply_rule(ps, rule.unsqueeze(0).to(M.DEV))[0]
    p = pred.permute(1,2,0).cpu().numpy(); t = tgt.to(M.DEV).permute(1,2,0).cpu().numpy()
    lp = cs.rgb_to_lab(p); lt = cs.rgb_to_lab(t)
    de = cs.delta_e_2000(lp, lt)                      # 逐像素 ΔE
    # 逐像素色相角误差（Lab a/b 平面），只统计有色像素（目标色度>10）
    hp = np.arctan2(lp[...,2], lp[...,1]); ht = np.arctan2(lt[...,2], lt[...,1])
    dh = np.abs(np.angle(np.exp(1j*(hp-ht)))) * 180/np.pi
    chroma_t = np.sqrt(lt[...,1]**2 + lt[...,2]**2)
    colored = chroma_t > 10
    print(f"[{task}] 逐像素ΔE: mean={de.mean():.2f} median={np.median(de):.2f} p90={np.percentile(de,90):.2f} max={de.max():.1f}")
    print(f"   ΔE>5占比={ (de>5).mean()*100:.1f}%  ΔE>10占比={(de>10).mean()*100:.1f}%  ΔE<2(对齐)占比={(de<2).mean()*100:.1f}%")
    print(f"   有色像素色相角误差: mean={dh[colored].mean():.1f}° median={np.median(dh[colored]):.1f}° p90={np.percentile(dh[colored],90):.1f}°")
    # 番茄区(目标红: a>20)预测成了什么
    red = (lt[...,1] > 20) & (lt[...,0] > 30)
    if red.sum() > 0:
        print(f"   目标红像素({red.sum()}个): 目标Lab均值={lt[red].mean(0).round(1)} 预测Lab均值={lp[red].mean(0).round(1)}")