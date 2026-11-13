# -*- coding: utf-8 -*-
"""V4Flash 专家路由重叠率 — 决定专家预取是否可行
记录每 token 每层激活专家, 算相邻 token 同层重叠率 + 相邻窗口重叠率
"""
import sys, time, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 记录每token每层专家激活
REC = []  # list of {token_idx: {lid: set(eids)}}
_cur = None

_orig_lazy = vs.V4LazyMoE.forward
def _rec_lazy(self, hidden_states, input_ids=None):
    global _cur
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    uids = indices.unique().tolist()
    if _cur is not None:
        _cur[self.lid] = set(uids)
    # 原逻辑 (从原函数复制, 保持行为)
    B, S, H = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.reshape(-1, H)
    if self.mtp:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_mtp_expert, e) for e in uids}
    else:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_expert, self.lid, e) for e in uids}
    out = torch.zeros(flat.shape[0], H, device=flat.device, dtype=torch.float32)
    for eid in uids:
        w1, s1, w2, s2, w3, s3 = futs[eid].result()
        hit = (indices == eid)
        tidx = hit.any(-1).nonzero().squeeze(-1)
        if tidx.numel() == 0:
            continue
        x = flat[tidx]
        g = vs.FP8_KERNEL.matmul_2d(x, w1, s1, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        u = vs.FP8_KERNEL.matmul_2d(x, w3, s3, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        gate = g.clamp(max=self.limit); up = u.clamp(min=-self.limit, max=self.limit)
        inter = torch.nn.functional.silu(gate) * up
        d = vs.FP8_KERNEL.matmul_2d(inter, w2, s2, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        w = (weights * hit).sum(-1)[tidx]
        out[tidx] += d * w.unsqueeze(-1)
    shared = self.shared_experts(residual).to(torch.float32)
    return (out.view(B, S, H) + shared).to(hidden_states.dtype)
vs.V4LazyMoE.forward = _rec_lazy

print('[eo] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)
eng.forward(ids, return_hidden=True)

# 跑 15 token, 记录专家激活
logits, cache, h = eng.forward(ids, return_hidden=True)
for n in range(15):
    _cur = {}
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    REC.append(_cur)
    _cur = None

# 计算重叠率
print(f'\n[eo] 记录 {len(REC)} token, 每 token {len(REC[0]) if REC else 0} 层', flush=True)
# 相邻 token 同层重叠率
tot = 0; overlap = 0
for t in range(1, len(REC)):
    for lid in set(REC[t-1].keys()) & set(REC[t].keys()):
        a = REC[t-1][lid]; b = REC[t][lid]
        if a and b:
            tot += len(a)
            overlap += len(a & b)
print(f'相邻token同层专家重叠: {overlap}/{tot} = {overlap/max(tot,1)*100:.1f}%', flush=True)
# 相邻窗口 (8层组) 重叠率
tot_w = 0; ovl_w = 0
for t in range(1, len(REC)):
    for g in range(6):
        lids = range(g*8, min(g*8+8, 43))
        exp_prev = set(); exp_cur = set()
        for lid in lids:
            exp_prev |= REC[t-1].get(lid, set())
            exp_cur |= REC[t].get(lid, set())
        if exp_prev and exp_cur:
            tot_w += len(exp_cur)
            ovl_w += len(exp_cur & exp_prev)
print(f'相邻token窗口(8层)专家重叠: {ovl_w}/{tot_w} = {ovl_w/max(tot_w,1)*100:.1f}%', flush=True)
# 跨token(隔1)同层重叠
tot2 = 0; ovl2 = 0
for t in range(2, len(REC)):
    for lid in set(REC[t-2].keys()) & set(REC[t].keys()):
        a = REC[t-2][lid]; b = REC[t][lid]
        if a and b:
            tot2 += len(a); ovl2 += len(a & b)
print(f'隔1token同层专家重叠: {ovl2}/{tot2} = {ovl2/max(tot2,1)*100:.1f}%', flush=True)
print('[eo] DONE', flush=True)
