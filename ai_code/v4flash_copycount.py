# -*- coding: utf-8 -*-
"""V4Flash copy 来源统计 — 复制 V4LazyMoE.forward 加计数, 量化每轮 copy 分布
统计: gather(flat[tidx]) / scatter(out[tidx]) / .to(dev) / matmul / 比较
"""
import sys, time, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import torch.nn.functional as F
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 计数器
CNT = collections.Counter()
BYTES = collections.Counter()

# 复制 V4LazyMoE.forward + 计数
_orig_lazy = vs.V4LazyMoE.forward
def _counted_lazy(self, hidden_states, input_ids=None):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import ACT2FN
    B, S, H = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.reshape(-1, H)
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)

    uids = indices.unique().tolist()
    if self.mtp:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_mtp_expert, e) for e in uids}
    else:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_expert, self.lid, e) for e in uids}
    out = torch.zeros(flat.shape[0], H, device=flat.device, dtype=torch.float32)
    for eid in uids:
        w1, s1, w2, s2, w3, s3 = futs[eid].result()
        hit_2d = (indices == eid)
        CNT['compare(==)'] += 1; BYTES['compare(==)'] += hit_2d.numel()
        tidx = hit_2d.any(-1).nonzero().squeeze(-1)
        CNT['nonzero'] += 1
        if tidx.numel() == 0:
            continue
        x = flat[tidx]
        CNT['gather_flat[tidx]'] += 1; BYTES['gather_flat[tidx]'] += x.numel() * x.element_size()
        dev = x.device
        w1g = w1.to(dev); s1g = s1.to(dev); w3g = w3.to(dev); s3g = s3.to(dev)
        w2g = w2.to(dev); s2g = s2.to(dev)
        CNT['w.to(dev)'] += 6; BYTES['w.to(dev)'] += (w1.numel()+w2.numel()+w3.numel()) * w1.element_size()
        g = vs.FP8_KERNEL.matmul_2d(x, w1g, s1g, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        u = vs.FP8_KERNEL.matmul_2d(x, w3g, s3g, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        CNT['matmul_2d'] += 2
        gate = g.clamp(max=self.limit)
        up = u.clamp(min=-self.limit, max=self.limit)
        inter = F.silu(gate) * up
        d = vs.FP8_KERNEL.matmul_2d(inter, w2g, s2g, [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        CNT['matmul_2d'] += 1
        w = (weights * hit_2d).sum(-1)[tidx]
        CNT['weight_mul_sum'] += 1
        out[tidx] += d * w.unsqueeze(-1)
        CNT['scatter_out[tidx]'] += 1; BYTES['scatter_out[tidx]'] += d.numel() * d.element_size()
    shared = self.shared_experts(residual).to(torch.float32)
    ret = (out.view(B, S, H) + shared).to(hidden_states.dtype)
    return ret
vs.V4LazyMoE.forward = _counted_lazy

print('[cc] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好", return_tensors='pt').input_ids.to(vs.DEVICE)

# 预热 + 跑 3 token 统计
logits, cache, h = eng.forward(ids, return_hidden=True)
for _ in range(3):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

print('\n===== Copy 来源统计 (3 token) =====', flush=True)
print(f"{'操作':<22} {'次数':>6} {'总MB':>8} {'每次KB':>8}", flush=True)
for op, cnt in CNT.most_common():
    mb = BYTES[op] / 2**20
    per = BYTES[op] / cnt / 1024 if cnt else 0
    print(f"{op:<22} {cnt:>6} {mb:>8.1f} {per:>8.1f}", flush=True)
print(f"\n总 copy 相关操作: {sum(CNT.values())} 次", flush=True)
print('[cc] DONE', flush=True)
