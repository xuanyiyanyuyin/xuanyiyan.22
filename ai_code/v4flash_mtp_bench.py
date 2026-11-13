# -*- coding: utf-8 -*-
"""MTP 模式基准 — 当前异步引擎 vs 同步备份引擎 (--backup 切换)
统计: wall / 主token / MTP额外token / acceptance rate / effective tok/s
"""
import sys, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')

ap = argparse.ArgumentParser()
ap.add_argument('--backup', action='store_true', help='用同步版备份引擎 (pre_d2h_async)')
ap.add_argument('--tokens', type=int, default=100, help='主 token 预算')
ap.add_argument('--prompt', type=str, default='请详细介绍量子计算的基本原理')
args = ap.parse_args()

if args.backup:
    import v4flash_chunked_sched_pre_d2h_async as vs
    tag = 'SYNC备份(无D2H异步)'
else:
    import v4flash_chunked_sched as vs
    tag = 'ASYNC(D2H异步化)'
print(f'[bench] 引擎: {tag}', flush=True)

import torch
from transformers import AutoTokenizer

print('[bench] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok(args.prompt, return_tensors='pt').input_ids.to(vs.DEVICE)
S0 = ids.shape[1]
print(f'[bench] prompt={S0} tokens', flush=True)

eng._mtp_stats = {'try': 0, 'accept': 0}
t0 = time.time()
out = eng.generate(ids, max_new_tokens=args.tokens, do_sample=True,
                   temperature=0.7, top_k=50, top_p=0.9)
wall = time.time() - t0

total = out.shape[1] - S0
acc = eng._mtp_stats['accept']
tr = eng._mtp_stats['try']
rate = acc / tr if tr else 0.0
main_tok = total - acc
eff = total / wall
print(f'[bench] === {tag} ===', flush=True)
print(f'[bench] wall={wall:.1f}s (含prefill) | 有效输出={total} tokens (主={main_tok} MTP额外={acc})', flush=True)
print(f'[bench] MTP 尝试={tr} 接受={acc} acceptance={rate:.1%}', flush=True)
print(f'[bench] effective={eff:.2f} tok/s | 主模型速度={main_tok/max(wall-1,0.1):.2f} tok/s', flush=True)
print(f'[bench] 输出前80字: {tok.decode(out[0][S0:S0+80], skip_special_tokens=True)!r}', flush=True)
print('[bench] DONE', flush=True)
