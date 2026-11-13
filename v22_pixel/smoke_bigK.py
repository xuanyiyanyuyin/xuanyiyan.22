# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T, v22_pixel_clean as M, v22_pixel_512 as V
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
src = V.load_src(SRC); src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
inp = inp.unsqueeze(0).to(T.DEV); rule = rule.unsqueeze(0).to(T.DEV)
for K in [512, 1024, 2048]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m = T.PixelEngineV22(dim=128, n_colors=K).to(T.DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        attr = m.encode_attr([(inp[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
        pred, logits, ct, ac, st, gate = m(inp, attr, rule)
        with torch.no_grad():
            sf = src_t.permute(0,2,3,1).reshape(-1,3); cb = m.color_codebook
            d = (sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t()
            ti = d.argmin(1).reshape(1,512,512)
        loss = F.cross_entropy(logits, ti)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        print(f"K={K}: OK 峰值VRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB CE={loss.item():.3f}")
        del m, opt, pred, logits, loss; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"K={K}: OOM 峰值VRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
        del m; torch.cuda.empty_cache()