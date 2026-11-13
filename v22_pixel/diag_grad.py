# -*- coding: utf-8 -*-
import os, sys, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"
src = V.load_src(SRC)
src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(T.DEV)
src_attr = torch.tensor(np.nan_to_num(M.P3Vis(chroma=True).process_image((src[:,:,::-1]*255).astype(np.uint8)),nan=0.,posinf=0.,neginf=0.),dtype=torch.float32).unsqueeze(0).to(T.DEV)
mask = (src_attr.abs().sum(0)>1e-6).float()
m = T.PixelEngineV22(dim=128).to(T.DEV)
inp, rule, tgt = V.make_pair(src, "color", np.random.default_rng(43))
attr = m.encode_attr([(inp.permute(1,2,0).numpy()*255).astype(np.uint8)[:,:,::-1]])
pred, ct, ac, st, gate = m(inp.unsqueeze(0).to(T.DEV), attr, rule.unsqueeze(0).to(T.DEV))
lab_p = M.torch_rgb_to_lab(pred); lab_s = M.torch_rgb_to_lab(src_t)
main_loss = F.l1_loss(lab_p, lab_s)
print("main_loss(Lab)=%.3f pred_range[%.3f,%.3f]" % (main_loss.item(), pred.min(), pred.max()))

def grad_report(tag):
    groups = {}
    for n, p in m.named_parameters():
        key = n.split('.')[0]
        if n.startswith('stages'): key = 'stages.' + n.split('.')[1]
        g = p.grad
        gn = float(g.norm()) if g is not None else -1
        groups.setdefault(key, []).append(gn)
    print(f"--- {tag} 逐模块梯度范数 ---")
    for k, v in groups.items():
        print(f"  {k:18s} sum={sum(v):.4e} max={max(v):.4e} n={len(v)}")

m.zero_grad()
main_loss.backward(retain_graph=True)
grad_report("仅主loss(Lab)")

# 完整 batch_loss
master = T.MasterGate().to(T.DEV)
lB = F.mse_loss(ct*mask, src_attr*mask).clamp(max=1.0)
lC = F.mse_loss(ac, src_attr[:,:48]).clamp(max=1.0)
lA = F.mse_loss(st, src_attr[:,48:112]).clamp(max=1.0)
gl = (gate.std()*10).clamp(0.01,1.0)
temps = master(torch.stack([gl.detach(),lA.detach(),lB.detach(),lC.detach(),main_loss.detach()]))
batch = main_loss + (temps[1]*lA+temps[2]*lB+temps[3]*lC) + temps[0]*gl
m.zero_grad(); master.zero_grad()
batch.backward()
grad_report("完整batch_loss")
print("temps", temps.detach().cpu().numpy().round(3).tolist())