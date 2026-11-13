# -*- coding: utf-8 -*-
"""V4Flash 性能基准: 输入(prefill)/输出(decode·MTP) token 速度 + 单token延迟 + 内存"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

RESULT = {}
print('[bench] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
RESULT['load_time_s'] = round(time.time() - t0, 1)
RESULT['mtp'] = eng.mtp is not None
RESULT['mtp_steps'] = int(getattr(eng, 'mtp_steps', 1))
print(f'[bench] 引擎加载完成 ({RESULT["load_time_s"]}s) MTP={"有" if eng.mtp else "无"} steps={RESULT["mtp_steps"]}', flush=True)

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)

# 固定长度 prompt (~128 token)
text = ("DeepSeek-V4-Flash 是一款高效的多专家混合架构大语言模型，"
        "采用了滑动窗口注意力与压缩稀疏注意力的混合设计，"
        "在保持高性能的同时大幅降低计算成本。"
        "该模型通过 MQA 共享键值机制减少内存占用，"
        "并使用分层缓存将权重分布在 GPU、内存与磁盘之间。"
        "推理引擎采用逐层预读取与专家按需加载策略，"
        "在有限显存下实现了长文本的高效生成。") * 3
ids = tok(text, return_tensors='pt').input_ids[:, :128].to(vs.DEVICE)
N = ids.shape[1]
RESULT['prompt_tokens'] = N
print(f'[bench] prompt {N} tokens', flush=True)

# 1. Prefill 计时
print('[bench] prefill...', flush=True)
t0 = time.time()
logits, cache = eng.forward(ids)
t_pre = time.time() - t0
RESULT['prefill_time_s'] = round(t_pre, 1)
RESULT['prefill_tok_s'] = round(N / t_pre, 1)
print(f'[bench] prefill {N} tok in {t_pre:.1f}s = {N/t_pre:.1f} tok/s', flush=True)

# 2. 单 token decode 延迟 (非MTP基线)
nxt = logits[:, -1, :].argmax(-1, keepdim=True)
t0 = time.time()
logits2, cache, h = eng.forward(nxt, past_key_values=cache, return_hidden=True)
t_dec1 = time.time() - t0
RESULT['single_decode_s'] = round(t_dec1, 1)
RESULT['single_decode_tok_s'] = round(1.0 / t_dec1, 2)
print(f'[bench] 单token decode {t_dec1:.1f}s = {1.0/t_dec1:.2f} tok/s', flush=True)

# 3. MTP generate (8 rounds)
print('[bench] generate max_new_tokens=8 (MTP)...', flush=True)
t0 = time.time()
out = eng.generate(ids, max_new_tokens=8, do_sample=False)
t_gen = time.time() - t0
n_gen = int(out.shape[1] - N)
RESULT['mtp_gen_time_s'] = round(t_gen, 1)
RESULT['mtp_gen_tokens'] = n_gen
RESULT['mtp_gen_tok_s'] = round(n_gen / t_gen, 2)
print(f'[bench] generate 产出 {n_gen} tok in {t_gen:.1f}s = {n_gen/t_gen:.2f} tok/s', flush=True)

# 4. 内存
import psutil
RESULT['ram_gb'] = round(psutil.Process().memory_info().rss / 2**30, 1)
RESULT['vram_alloc_gb'] = round(torch.cuda.memory_allocated() / 2**30, 2)
RESULT['vram_reserved_gb'] = round(torch.cuda.memory_reserved() / 2**30, 2)
print(f'[bench] RAM {RESULT["ram_gb"]}GB / VRAM alloc {RESULT["vram_alloc_gb"]}GB reserved {RESULT["vram_reserved_gb"]}GB', flush=True)

# 5. 输出样例 (仅参考)
sample = tok.decode(out[0, N:], skip_special_tokens=True)
RESULT['sample'] = sample[:120]
print(f'[bench] 输出样例: {sample[:120]!r}', flush=True)

print('[bench] RESULT=' + json.dumps(RESULT, ensure_ascii=False), flush=True)
print('[bench] DONE', flush=True)
