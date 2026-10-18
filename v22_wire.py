# -*- coding: utf-8 -*-
"""
V22.1 组件接线层 (v22_wire)
============================================
把 v22_vmf_modules.py 里"已写代码但未接入训练循环"的 10 个组件真正接进
train_v22_stage1.py，并提供统一的安全回退与日志。

接入清单（对应架构文档 §V22.1）:
  1. KappaPhaseScheduler        — κ 相变调度（每 step 推进；转相时触发一次 κ boost）
  2. CompetitiveCrystallization — 高密度区赢家结晶（每 density_window 步）
  3. SphericalVoronoiProbe      — 覆盖熵（每 probe_every 步）
  4. LocalBettiMonitor          — β₀ 拓扑监控（每 probe_every 步）
  5. PCACurvatureProxy          — 动量曲率代理（每 probe_every 步）
  6. LocalCurvatureEstimator    — 多尺度曲率 + 自适应阈值（每 step）
  7. MomentumExchangePool       — 7-Adam 松耦合预警（每 probe_every 步）
  8. AdaptiveProbeSet           — 探针重分布（每 probe_every 步，含 info gain 日志）
  9. ActiveExperimentProbe      — 因果矩阵（每 causal_every 步，一次扰动）
 10. SSOL + MutualInformationPenalty — 反冗余正则（每 step，加进 loss，权重可调）

微头定义:
  explore.net[-1].weight 形状 [256, 512]，256 行 = 256 个门控维度的输入方向。
  这与 meta.kappa (dim=256) 一一对应，因此 "微头 i" 的方向 = 该行归一化向量。

安全:
  每个组件调用都包在 _safe() 里，任何异常只记一次日志、不打断训练。
"""
import math
import os
import torch
import torch.nn.functional as F


def _safe(fn, default=None, tag=""):
    try:
        return fn()
    except Exception as e:
        print(f"  [WIRE] {tag} 失败(已忽略): {type(e).__name__}: {e}", flush=True)
        return default


class V22Wire:
    def __init__(self, explore, meta, master=None, reg_weight=0.005,
                 probe_every=50, causal_every=250, density_window=50,
                 k_boost=1.6, kappa_cap=6.0, total_steps=2000, kboost_enable=None):
        self.explore = explore
        self.meta = meta
        self.master = master
        self.reg_weight = reg_weight
        self.probe_every = probe_every
        self.causal_every = causal_every
        self.density_window = density_window
        self.k_boost = k_boost
        self.kboost_enable = (os.environ.get("WIRE_KBOOST", "0") == "1") if kboost_enable is None else bool(kboost_enable)
        self.kappa_cap = kappa_cap
        self.total_steps = max(int(total_steps), 1)

        import v22_vmf_modules as M
        self.M = M
        n_heads = int(meta.kappa.numel())
        self.n_heads = n_heads
        self.head_dim = int(explore.net[-1].weight.shape[1])

        self.sched = _safe(lambda: M.KappaPhaseScheduler(total_steps=self.total_steps, explore_ratio=0.4), tag="KappaPhaseScheduler")
        self.cryst = _safe(lambda: M.CompetitiveCrystallization(n_heads=n_heads, density_window=density_window), tag="CompetitiveCrystallization")
        self.voronoi = _safe(lambda: M.SphericalVoronoiProbe(n_probes=50, n_heads=n_heads, head_dim=self.head_dim), tag="SphericalVoronoiProbe")
        self.betti = _safe(lambda: M.LocalBettiMonitor(n_heads=n_heads, k_neighbors=8), tag="LocalBettiMonitor")
        self.pca = _safe(lambda: M.PCACurvatureProxy(window=50, warn_threshold=0.85), tag="PCACurvatureProxy")
        self.curv = _safe(lambda: M.LocalCurvatureEstimator(short_window=10, long_window=50), tag="LocalCurvatureEstimator")
        self.pool = None  # 惰性: 第一次 post_step 拿到 opts 再建
        self.probes = _safe(lambda: M.AdaptiveProbeSet(n_probes=50, probe_dim=self.head_dim), tag="AdaptiveProbeSet")
        self.active = _safe(lambda: M.ActiveExperimentProbe(n_heads=n_heads, output_dim=n_heads), tag="ActiveExperimentProbe")
        self.ssol = _safe(lambda: M.SphericalSelfOrganizingLoss(repulsion_strength=0.01, min_distance=0.1), tag="SSOL")
        self.mi = _safe(lambda: M.MutualInformationPenalty(penalty_weight=0.005, high_sim_threshold=0.85), tag="MI")

        if self.probes is not None:
            self.probes.refine_every = max(50, self.total_steps // 10)   # 短跑也能触发重分布
        self._act_hist = []
        self._betti_last = (0, 0)   # (step, reorg) 供窗口速率
        self._last_phase = None
        self._k_boost_count = 0
        self._step = 0

    # ───────────────────────── 微头方向 ─────────────────────────
    def head_vectors(self):
        """[n_heads, head_dim] 单位向量；来自 explore 最后一层的行方向。"""
        w = self.explore.net[-1].weight
        return F.normalize(w, dim=-1, eps=1e-8)

    # ───────────────────────── 反冗余正则 ─────────────────────────
    def regularizer(self, hv=None):
        if hv is None:
            hv = self.head_vectors()
        loss = None
        if self.ssol is not None:
            v = _safe(lambda: self.ssol(hv), tag="SSOL.forward")
            if v is not None:
                loss = v if loss is None else loss + v
        if self.mi is not None:
            v = _safe(lambda: self.mi(hv), tag="MI.forward")
            if v is not None:
                loss = v if loss is None else loss + v
        return loss

    # ───────────────────────── κ 相变 ─────────────────────────
    def _step_kappa(self, gates_mean):
        if self.sched is None:
            return {}
        _safe(lambda: self.sched.step(), tag="sched.step")
        phase = getattr(self.sched, "phase", "?")
        temp = _safe(lambda: self.sched.get_temperature(), tag="sched.temperature")
        if temp is not None:
            temp = max(0.1, min(1.0, float(temp)))   # 防 crystallize progress>1 时负温度
        info = {"phase": phase, "temp": temp}
        if self.kboost_enable and phase != self._last_phase and phase == "crystallize":
            # 转相: 用高激活的微头作为赢家, 把它们的 κ 抬一档
            mask = None
            # mask 来源: Voronoi/覆盖度 top-k 密度 (对应架构 :117 "Voronoi 密度稳定")
            mask = None
            hv_now = _safe(self.head_vectors, tag="boost.hv")
            if hv_now is not None:
                def _dense_mask():
                    h = hv_now.detach().float().cpu()
                    n_p = min(50, h.shape[0])
                    probs = F.normalize(h[:n_p], dim=-1)
                    sims = probs @ h.t()                     # [n_p, n_heads]
                    dens = sims.topk(5, dim=-1).values.mean(dim=-1)
                    thr = dens.mean() + dens.std()
                    return (dens > thr)
                mask = _safe(_dense_mask, tag="boost.mask")
            def do_boost():
                if mask is None:
                    self.meta.trigger_phase_transition(torch.ones(self.n_heads, dtype=torch.bool, device=self.meta.kappa.device), alpha_boost=1.5)
                else:
                    self.meta.trigger_phase_transition(mask.to(self.meta.kappa.device), alpha_boost=self.k_boost)
                with torch.no_grad():
                    self.meta.kappa.clamp_(max=self.kappa_cap)
            _safe(do_boost, tag="trigger_phase_transition")
            self._k_boost_count += 1
        self._last_phase = phase
        return info

    # ───────────────────────── 覆盖度(top-k) ─────────────────────────
    def _coverage_topk(self, hv, k=5):
        """512 维下 cos>0.5 不可达: 用每探针 top-k 相似度做密度, 再算熵."""
        import math as _m
        h = hv.detach().float().cpu()
        n_probe = 50
        probs = F.normalize(h[:n_probe], dim=-1) if h.shape[0] >= n_probe else F.normalize(torch.randn(n_probe, h.shape[1]), dim=-1)
        sims = probs @ h.t()                      # [n_probe, n_heads]
        dens = sims.topk(min(k, sims.shape[1]), dim=-1).values.mean(dim=-1)
        dens = dens / (dens.sum() + 1e-8)
        ent = float(-(dens * (dens + 1e-8).log()).sum())
        return {"entropy": round(ent, 4), "max_entropy": round(_m.log(n_probe), 4), "topk_mean_sim": round(float(sims.topk(min(k, sims.shape[1]), dim=-1).values.mean()), 4)}

    # ───────────────────────── 每步后处理 ─────────────────────────
    def post_step(self, step, gates=None, loss_val=None, grad_norm=None, opts=None):
        self._step += 1
        s = int(step)
        hv = _safe(self.head_vectors, tag="head_vectors")
        gates_mean = None
        if gates is not None:
            gates_mean = _safe(lambda: gates.detach().float().mean(dim=0), tag="gates_mean")
            if gates_mean is not None:
                self._act_hist.append(gates_mean.cpu())
                if len(self._act_hist) > max(self.density_window, 100):
                    self._act_hist.pop(0)

        info = self._step_kappa(gates_mean)

        # 曲率估计 (每步, 便宜)
        if self.curv is not None and grad_norm is not None:
            _safe(lambda: self.curv.step(float(grad_norm)), tag="curv.step")
            def _curv_est():
                try:
                    return self.curv.estimate()
                except Exception:
                    import torch as _t
                    s = _t.tensor(list(self.curv.short_history), dtype=_t.float32)
                    l = _t.tensor(list(self.curv.long_history), dtype=_t.float32)
                    if len(s) < 5: return (0.0, self.curv.threshold)
                    c = float(s.std() / (s.mean() + 1e-8)) / float(l.std() / (l.mean() + 1e-8) + 1e-8)
                    return (c, self.curv.threshold)
            info["curv"] = _safe(_curv_est, tag="curv.estimate")
            info["curv_thr"] = _safe(lambda: self.curv.step_threshold(self.total_steps, min(s, self.total_steps)), tag="curv.thr")

        if self.pool is None and opts is not None:
            self.pool = _safe(lambda: self.M.MomentumExchangePool(n_modules=len(opts)), tag="MomentumExchangePool")

        # 昂贵的监控: 每 probe_every 步
        if s % self.probe_every == 0 and hv is not None:
            if self.voronoi is not None:
                _safe(lambda: self.voronoi.step(hv.detach().cpu()), tag="voronoi.step")
                info["coverage"] = _safe(lambda: self.voronoi.get_coverage_stats(), tag="voronoi.stats")
                info["coverage_topk"] = _safe(lambda: self._coverage_topk(hv), tag="coverage.topk")
            if self.betti is not None:
                _safe(lambda: self.betti.step(hv.detach().cpu()), tag="betti.step")
                _st = _safe(lambda: self.betti.get_status(), tag="betti.status")
                # 累计计数器 → 窗口速率（否则"一直在涨"是构造必然）
                _cur = getattr(self.betti, "reorg_events", None)
                if _st is not None and _cur is not None:
                    _ps, _pv = self._betti_last
                    _d = _cur - _pv; _ds = max(1, s - _ps)
                    _st = dict(_st) if isinstance(_st, dict) else {"status": _st}
                    _st["reorg_rate_per_1k"] = round(_d * 1000.0 / _ds, 1)
                    self._betti_last = (s, _cur)
                info["betti"] = _st
            if self.pca is not None and self.meta.kappa.grad is not None:
                _safe(lambda: self.pca.step(self.meta.kappa.grad.detach().clone().flatten()), tag="pca.step")
            if self.probes is not None and hv is not None:
                hv_c = hv.detach().cpu()
                _pg = _safe(lambda: self.probes.compute_info_gain(hv_c), tag="probe.gain")
                info["probe_gain"] = (float(_pg.mean()), float(_pg.std())) if _pg is not None else None
                _safe(lambda: self.probes.refine(hv_c), tag="probe.refine")
            if self.cryst is not None and gates_mean is not None:
                _safe(lambda: self.cryst.record_activations(gates_mean.detach().cpu()), tag="cryst.record")
                regions = _safe(lambda: self.cryst.detect_high_density_regions(), default=[], tag="cryst.detect")
                info["cryst_regions"] = len(regions) if isinstance(regions, list) else -1
            if self.pool is not None and opts is not None and grad_norm is not None:
                for i in range(len(opts)):
                    _safe(lambda i=i: self.pool.broadcast(i, float(grad_norm)), tag="pool.broadcast")
                info["pool_warn"] = _safe(lambda: [self.pool.query(i, 0.3) for i in range(len(opts))], tag="pool.query")

        # 因果矩阵: 每 causal_every 步做一次受控扰动
        if s % self.causal_every == 0 and gates_mean is not None:
            gm = gates_mean.detach()
            idx = int(torch.argmax(gm).item())
            gm_c = gm.detach().cpu()
            pert = _safe(lambda: self.active.inject_perturbation(gm_c, idx), tag="active.inject")
            if pert is not None:
                delta = _safe(lambda: (pert - gm_c).float(), tag="active.delta")
                if delta is not None:
                    _safe(lambda: self.active.update_causal_matrix(delta, idx), tag="active.update")
            info["top_heads"] = _safe(lambda: self.active.get_top_influential_heads(k=5), tag="active.top")

        info["kappa"] = _safe(lambda: dict(self.meta.get_kappa_stats()), tag="kappa.stats")
        info["reg"] = None
        if s % 100 == 0:
            print(f"  [WIRE] step={s} " + " ".join(
                f"{k}={v}" for k, v in info.items()
                if k in ("phase", "temp", "coverage", "betti", "kappa", "probe_gain", "cryst_regions", "curv")
            ), flush=True)
        return info


class _NullWire:
    """WIRE_DISABLE=1 时的空接线（clean control 用：不构造任何组件，不消耗 RNG）。"""
    reg_weight = 0.0
    def head_vectors(self):
        return None
    def regularizer(self, hv=None):
        return None
    def post_step(self, *a, **k):
        return {}
