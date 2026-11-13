# -*- coding: utf-8 -*-
"""双盘分布诊断 — 跑 90 token, 统计专家读取的 C盘/D盘 分布
判断: 双盘没并行读? 还是激活专家分布不均导致某盘覆盖率低?
"""
import sys, time, os, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# C 盘 shard 集合
C_SHARDS = set()
if os.path.exists(r'C:/ai_v4flash_experts'):
    C_SHARDS = set(s for s in os.listdir(r'C:/ai_v4flash_experts') if s.endswith('.safetensors'))

STAT = {'n': 0, 'c': 0, 'd': 0, 'c_unique': set(), 'd_unique': set()}
# 每 token 统计
TOK = []
_cur = None

_orig_read = vs.ExpertReader.read_expert
def _timed_read(self, lid, eid):
    shard = self.disk.get(f'layers.{lid}.ffn.experts.{eid}.w1.weight', '')
    is_c = shard in C_SHARDS
    STAT['n'] += 1
    if is_c:
        STAT['c'] += 1; STAT['c_unique'].add((lid, eid))
    else:
        STAT['d'] += 1; STAT['d_unique'].add((lid, eid))
    if _cur is not None:
        _cur['c' if is_c else 'd'] += 1
    return _orig_read(self, lid, eid)
vs.ExpertReader.read_expert = _timed_read

print(f'[dd] C盘 shards: {len(C_SHARDS)}/46 ({"双盘" if C_SHARDS else "单盘"})', flush=True)
print('[dd] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

# 预热 10 token (页缓存热)
_l, _c, _h = logits, cache, h
for i in range(10):
    A = _l[:, -1, :].argmax(-1, keepdim=True)
    _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
logits, cache, h = _l, _c, _h
print('[dd] 预热完成, 统计 90 token...', flush=True)

# 90 token 统计
for i in range(90):
    _cur = {'c': 0, 'd': 0}
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    TOK.append(_cur)
    if i % 15 == 14:
        print(f'[dd]   ...{i+1} token done', flush=True)

# 汇总
tot_c = sum(t['c'] for t in TOK); tot_d = sum(t['d'] for t in TOK)
n_tok = len(TOK)
print(f'[dd] === 90 token 统计 ===', flush=True)
print(f'[dd] 专家读取: 总={STAT["n"]} | C盘={tot_c}({tot_c/max(tot_c+tot_d,1)*100:.0f}%) D盘={tot_d}({tot_d/max(tot_c+tot_d,1)*100:.0f}%)', flush=True)
print(f'[dd] 每token平均: C={tot_c/n_tok:.0f} D={tot_d/n_tok:.0f} 总={(tot_c+tot_d)/n_tok:.0f}', flush=True)
print(f'[dd] unique专家: C盘={len(STAT["c_unique"])} D盘={len(STAT["d_unique"])}', flush=True)
# 覆盖率: C盘23shard有多少专家?
c_total_exp = len(C_SHARDS) * 256  # 近似每shard 256专家
d_total_exp = (46 - len(C_SHARDS)) * 256
print(f'[dd] C盘可用专家≈{c_total_exp} D盘≈{d_total_exp}', flush=True)
print(f'[dd] C盘覆盖率={len(STAT["c_unique"])/max(c_total_exp,1)*100:.0f}% D盘覆盖率={len(STAT["d_unique"])/max(d_total_exp,1)*100:.0f}%', flush=True)
# 每 token 双盘是否同时活跃
both = sum(1 for t in TOK if t['c'] > 0 and t['d'] > 0)
print(f'[dd] 双盘同时活跃的token: {both}/{n_tok} ({both/n_tok*100:.0f}%)', flush=True)
print('[dd] DONE', flush=True)
