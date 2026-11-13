# -*- coding: utf-8 -*-
"""H3 手写引擎的对齐打分器（不动引擎，只借它的函数）。

给模型一段【有标准答案】的输入：把参考图 latent 平铺成 7 帧当 x0（合法的静态视频），
按 σ 加噪成 x_σ = (1-σ)x0 + σ·eps。官方约定 x0 = x_σ + σ·v ⇒ 真值速度 v* = x0 - eps。

    rel = MSE(v_pred, v*) / MSE(0, v*)

rel = 1.0 ⇒ 模型等于啥也没预测；明显低于 1 ⇒ 前向与权重是对齐的。
换 S/T/U/P 任何一处都用同一个数比，不用等 11 分钟出片。
用法: python h3_score.py [参考图] [sigma] [turbo=4|none]
"""
import os, sys, time, torch
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\ai")
import h3_engine as E

AUD_CH, AUD_LC = 2, 32      # components.audio_channels / audio_latent_channels

def build_packed(ref, sigma, turbo="none"):
    """造与官方 ref2va 同构的打包输入，供打分与对齐台共用。"""
    H = W = 512
    if turbo != "none":
        E.load_turbo(E.TURBO_PATHS[turbo])
    r = E.Reader()
    sm = r.small()
    lf = E.auto_slice(r, resident=3, cache=6, prefetch=True)
    v, c = E._vae()
    lat = E.encode_ref(v, c, ref)                       # (1,24,1,h,w) 归一化
    xv_ref = E.patchify(lat).to(E.DEV).float()
    Nr = xv_ref.shape[1]
    n_lat = 7
    xv0 = E.patchify(lat.repeat(1, 1, n_lat, 1, 1)).to(E.DEV).float()
    g = torch.Generator(device=E.DEV).manual_seed(7)
    eps = torch.randn(xv0.shape, generator=g, device=E.DEV)
    tgt = (1.0 - sigma) * xv0 + sigma * eps
    # [FIX-1017 SCORE-COND] 条件行要按官方 noise-augment 到 t=0.999，不能喂全干净：
    #   before_denoise.py:968-970 "the released model noises them to `t = 0.999` and holds them there for every step"
    #   → scheduler.scale_noise(condition, keyframe_noise_aug, noise)，而 scale_noise 是
    #   `t*sample + (1-t)*noise`（scheduling_minimax_h3.py:193-221，timestep 按字面取，不查表）
    #   ⇒ 0.999*x0_ref + 0.001*noise。旧代码喂全干净 latent 却标 t=0.999，属于分布外输入。
    _g42 = torch.Generator(device=E.DEV).manual_seed(42)
    xv_ref = 0.999 * xv_ref + 0.001 * torch.randn(xv_ref.shape, generator=_g42, device=E.DEV)
    xt = torch.cat([xv_ref, tgt], dim=1)
    Nv = xt.shape[1]

    cd = torch.load(os.environ.get("H3_COND_PT", "D:/ai/h3_cond_official.pt"),
                    map_location="cpu", weights_only=True)
    cond = cd["hidden_states"].float()
    if cond.dim() == 2:
        cond = cond.unsqueeze(0)
    cond = cond.to(E.DEV)
    ttags = cd["token_tags"].reshape(-1).to(torch.long)
    Nt = int(ttags.shape[0])
    n_aud = int(round(22 / 24.0 * 40)); na = n_aud * AUD_CH
    xa = torch.randn(1, na, AUD_LC, device=E.DEV)
    L = Nv + Nt + na
    vid = torch.arange(0, Nv, device=E.DEV)
    txt = torch.arange(Nv, Nv + Nt, device=E.DEV)
    aud = torch.arange(Nv + Nt, L, device=E.DEV)
    tags = torch.cat([torch.zeros(Nv, dtype=torch.long), ttags.cpu(),
                      torch.full((na,), 2, dtype=torch.long)]).to(E.DEV)

    pos = torch.zeros(L, 3, dtype=torch.float64)
    pos[Nv:Nv + Nt, 0] = torch.arange(Nt, dtype=torch.float64)
    _ph = _pw = 2
    rotary_time = float(Nt)
    _rh, _rw = int(lat.shape[-2]), int(lat.shape[-1])
    _ref_grid, _ = E._frame_position_grid(_rh, _rw, _ph, _pw)
    pos[0:Nr, 0] = rotary_time
    pos[0:Nr, 1:] = _ref_grid
    rotary_time += 1.0
    _tg, _tw = E._frame_position_grid(H // 16, W // 16, _ph, _pw)
    _rpf = _tg.shape[0]
    pos[Nr:Nv, 0] = E._temporal_position_grid(n_lat, rotary_time).repeat_interleave(_rpf)
    pos[Nr:Nv, 1:] = _tg.repeat(n_lat, 1)
    E._fill_audio_positions(pos, slice(Nv + Nt, L), n_aud, rotary_time, _tw, AUD_CH)
    pos = pos.to(E.DEV)

    _t = 1.0 - sigma
    row_t = torch.full((L,), _t, dtype=torch.float32, device=E.DEV)
    row_t[:Nr] = 0.999                                   # 官方条件行锚点
    # [FIX-1017 SCORE-AUDIO] 音频行不能再挂视频的时刻表。官方是两条调度耦合在同一根 base 网格上：
    #   σ_a = time_shift_sigma(σ_v, 12→3)（ComfyUI model.py:35-38 的闭式；σ_v=0.2 ⇒ σ_a=0.0588）。
    #   打分器没有干净音频可插值，于是取"自洽"的那一端：xa 是纯 randn（σ_a=1）⇒ 时刻必须标 t=0，
    #   这才是分布内输入（等于真实采样第 0 步的音频流）。旧代码把纯噪声标成 t=0.8（近乎干净），
    #   74 行分布外 token 会污染视频行的注意力，是 rel 下不去的一个已知来源。
    #   代价：视频在 σ、音频在 σ=1，两者不同步（真实跑时不会这样）⇒ rel 只能当相对裁判，不是绝对误差。
    row_t[Nv + Nt:] = 0.0
    uniq, ti = torch.unique(row_t, sorted=True, return_inverse=True)
    print("[s] σ=%.3f t=%.3f  序列 L=%d（视频 %d=参考 %d+目标 %d / 文本 %d / 音频 %d）" % (
        sigma, _t, L, Nv, Nr, Nv - Nr, Nt, na), flush=True)
    return dict(r=r, sm=sm, lf=lf, xt=xt, xa=xa, cond=cond, uniq=uniq, ti=ti, tags=tags,
                pos=pos, vid=vid, aud=aud, txt=txt, xv0=xv0, eps=eps, Nr=Nr, sigma=sigma, tgt=tgt)

def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else r"D:\ai\output\images\gen_68965_0.png"
    sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    turbo = sys.argv[3] if len(sys.argv) > 3 else "none"
    P = build_packed(ref, sigma, turbo)
    r, sm, lf = P["r"], P["sm"], P["lf"]
    xt, xa, cond = P["xt"], P["xa"], P["cond"]
    uniq, ti, tags, pos = P["uniq"], P["ti"], P["tags"], P["pos"]
    vid, aud, txt = P["vid"], P["aud"], P["txt"]
    Nr, xv0, eps, tgt, sigma = P["Nr"], P["xv0"], P["eps"], P["tgt"], P["sigma"]
    t0 = time.perf_counter()
    vo, ao = E.dit_forward(r, xt, xa, cond, uniq, ti, tags, pos, vid, aud, txt, sm=sm, layer_fn=lf)
    vs = xv0 - eps
    base = float((vs ** 2).mean())
    vp = vo[:, Nr:].float()
    print("[s] %.0fs  |v_pred|std=%.3f |v*|std=%.3f" % (
        time.perf_counter() - t0, float(vp.std()), float(vs.std())), flush=True)
    # 一次前向同时检验"模型输出到底是什么"：速度 / 反号 / x0 预测 / eps 预测 / 只差一个尺度。
    # 若某个变体明显更低，说明错的是输出约定，不是权重。
    hyps = [
        ("v = vo（官方 data-ward 速度）", vp),
        ("v = -vo（noise-ward）", -vp),
        ("v = (vo - x)/σ（vo 当 x0 预测）", (vp - tgt) / sigma),
        ("v = (x - vo)/σ", (tgt - vp) / sigma),
        ("v = vo/σ", vp / sigma),
        ("v = vo·(1-σ)/σ", vp * (1.0 - sigma) / sigma),
        ("v = x0(由eps反解) - vo（vo 当 eps 预测）", (tgt - sigma * vp) / (1.0 - sigma) - vp),
    ]
    for name, h in hyps:
        print("[s]   rel=%.4f  %s" % (float(((h - vs) ** 2).mean()) / base, name), flush=True)
    print("[s] ★★★ 采用官方约定时 rel = %.4f（1.0=啥也没预测；基线 MSE=%.4f）" % (
        float(((vp - vs) ** 2).mean()) / base, base), flush=True)
    return True

if __name__ == "__main__":
    main()
    os._exit(0)
