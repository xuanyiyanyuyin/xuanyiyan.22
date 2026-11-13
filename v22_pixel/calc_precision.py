# -*- coding: utf-8 -*-
"""纯计算：字典粒度K决定的精度下限（K-means量化ΔE），不训练。"""
import os, sys, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import color_science as cs, v22_pixel_512 as V
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
src = V.load_src(SRC)
lab = cs.rgb_to_lab(src)
pix = lab.reshape(-1, 3).astype(np.float32)
rng = np.random.default_rng(0)
fit = pix[rng.choice(len(pix), 60000, replace=False)]
ev = pix[rng.choice(len(pix), 20000, replace=False)].reshape(200, 100, 3)
print("K | 量化ΔE00下限 | ΔE<2占比 | ΔE<5占比")
for K in [64, 128, 256, 512, 1024, 2048, 4096]:
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(fit, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    evf = ev.reshape(-1, 3)
    d = ((evf[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    q = centers[d.argmin(1)].reshape(200, 100, 3)
    de = cs.delta_e_2000(q, ev)
    print(f"{K:5d} | {de.mean():8.3f} | {(de<2).mean()*100:6.1f}% | {(de<5).mean()*100:6.1f}%")