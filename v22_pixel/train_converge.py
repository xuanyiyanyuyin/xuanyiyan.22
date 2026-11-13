# -*- coding: utf-8 -*-
"""V22 像素引擎收敛驱动：断点续训＋逐段加轮数＋收敛状态＋平台检测。
自动任务每次跑 --add_epochs 轮，读 converge_state.json 续训，量 color CE，判平台。
收敛靶：color 平均 CE ≲ 0.5（逐像素 CE<0.1→ΔE2.23 字典下限）。平台期查 GitHub V22 + 经验库。"""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v22_pixel_v22full as T
import v22_pixel_clean as M
import v22_pixel_512 as V
import color_science as cs

DEV = T.DEV
SRC = r"C:\Users\Jinqi Wei\.qoder-cn\tmp\D--xianliao\images\e057dd71-ef5f-4923-a60a-6ecc31af5fbd\cdb5de12-5afb-4c49-9bf7-96124b8a3a5e.png"


def color_metrics(m, src, src_t, cinp, crule):
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
        aligned = (cs.delta_e_2000(cs.rgb_to_lab(pn), cs.rgb_to_lab(src)) < 2).mean()
    m.train(); return ce, float(de), float(aligned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add_epochs", type=int, default=500)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n_colors", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "pixel_converge"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42); rng = np.random.default_rng(42)
    torch.cuda.set_per_process_memory_fraction(1.0)
    src = V.load_src(SRC); src_t = torch.tensor(src).permute(2,0,1).unsqueeze(0).to(DEV)
    state_path = os.path.join(args.out, "converge_state.json")
    ckpt_path = os.path.join(args.out, "model.pt")
    state = {"total_epochs": 0, "ce_history": [], "de_history": [], "plateau": False, "best_ce": 99.0}
    if os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))
    m = T.PixelEngineV22(dim=args.dim, n_colors=args.n_colors).to(DEV)
    master = T.MasterGate().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    opt_master = torch.optim.Adam(master.parameters(), lr=args.lr*0.5)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=True)
        m.load_state_dict(ck["model"]); master.load_state_dict(ck["master"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        print(f"[续训] 从 E{state['total_epochs']} 续，best_ce={state['best_ce']:.3f}", flush=True)
    tasks = ["color","shift","flip","combo","denoise"]
    cinp, crule, _ = V.make_pair(src, "color", np.random.default_rng(43))
    cinp = cinp.unsqueeze(0).to(DEV); crule = crule.unsqueeze(0).to(DEV)
    ce_ema, aux_val, cool_down = 8.0, 1.0, 0
    start = state["total_epochs"]; t0 = time.time()
    for ep in range(1, args.add_epochs+1):
        # 平台期余弦退火(只降不升): 经验库13"lr高会振荡"+主人SelfLR loss升→lr降方向; 提lr分层已实测失稳(E4900→E5000 CE 0.502→0.968)
        lr_now = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * ep / args.add_epochs)))
        for pg in opt.param_groups: pg['lr'] = lr_now
        for pg in opt_master.param_groups: pg['lr'] = lr_now * 0.5
        inp, rule, tgt = V.make_pair(src, tasks[(ep-1)%5], rng)
        inp=inp.unsqueeze(0).to(DEV); rule=rule.unsqueeze(0).to(DEV)
        attr = m.encode_attr([(inp[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)[:,:,::-1]])
        pred, logits, ct, ac, st, gate = m(inp, attr, rule)
        with torch.no_grad():
            sf=src_t.permute(0,2,3,1).reshape(-1,3); cb=m.color_codebook
            d=(sf*sf).sum(1,keepdim=True)+(cb*cb).sum(1)[None,:]-2*sf@cb.t(); ti=d.argmin(1).reshape(1,512,512)
        main_loss=F.cross_entropy(logits,ti)
        lab_p=M.torch_rgb_to_lab(pred); lab_s=M.torch_rgb_to_lab(src_t)
        loss_C=F.l1_loss(lab_p[:,1:3],lab_s[:,1:3])/128.0
        Cp=torch.sqrt(lab_p[:,1]**2+lab_p[:,2]**2+1e-6); Cs=torch.sqrt(lab_s[:,1]**2+lab_s[:,2]**2+1e-6)
        loss_A=F.l1_loss(Cp,Cs)/128.0
        loss_B=1.0-M.torch_ssim(pred,src_t)
        gate_loss=(gate.std()*10).clamp(0.01,1.0)
        temps=master(torch.stack([gate_loss.detach(),loss_A.detach(),loss_B.detach(),loss_C.detach(),main_loss.detach()]))
        if main_loss.item()>ce_ema*3.0:
            cool_down=max(cool_down,50); main_loss=main_loss.clamp(max=ce_ema*3.0)
        ce_ema=0.90*ce_ema+0.10*main_loss.item()
        if cool_down>0: aux_val=max(0.2,aux_val*0.95); cool_down-=1
        else: aux_val=min(1.0,aux_val*1.01)
        batch=main_loss+aux_val*(temps[1]*loss_A+temps[2]*loss_B+temps[3]*loss_C)+temps[0]*gate_loss
        opt.zero_grad(); opt_master.zero_grad(); batch.backward()
        if not any(p.grad is not None and torch.isnan(p.grad).any() for p in m.parameters()):
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt_master.step()
        else: opt.zero_grad(); opt_master.zero_grad()
        if ep%100==0:
            ce,de,al=color_metrics(m,src,src_t,cinp,crule)
            state["ce_history"].append([start+ep,round(ce,4)]); state["de_history"].append([start+ep,round(de,3)])
            state["best_ce"]=min(state["best_ce"],ce)
            print(f"[E{start+ep}] colorCE={ce:.3f} ΔE={de:.2f} 对齐{al*100:.0f}% {time.time()-t0:.0f}s", flush=True)
            del pred,logits,ct,ac,st,gate,batch,main_loss,lab_p,lab_s; torch.cuda.empty_cache()
    state["total_epochs"]=start+args.add_epochs
    h=[c for _,c in state["ce_history"][-4:]]
    state["plateau"]= len(h)>=3 and (max(h)-min(h))<0.05
    json.dump(state, open(state_path,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    torch.save({"model":m.state_dict(),"master":master.state_dict(),"opt":opt.state_dict()}, ckpt_path)
    print(f"[段完成] 总E{state['total_epochs']} 当前CE={h[-1] if h else '?'} best={state['best_ce']:.3f} 平台={state['plateau']}", flush=True)
    print(f"[收敛靶] CE≲0.5；当前{h[-1] if h else '?'}；平台期查GitHub V22+经验库", flush=True)


if __name__ == "__main__":
    main()