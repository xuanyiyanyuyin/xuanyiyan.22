# -*- coding: utf-8 -*-
"""prefill 隔离测试: 只测 128 token 的 prefill 时间"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
import torch
from transformers import AutoTokenizer

print('[pf] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
print(f'[pf] 引擎加载 {time.time()-t0:.0f}s', flush=True)

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
text = ("DeepSeek-V4-Flash 是一款高效的多专家混合架构大语言模型，"
        "采用了滑动窗口注意力与压缩稀疏注意力的混合设计，"
        "在保持高性能的同时大幅降低计算成本。"
        "该模型通过 MQA 共享键值机制减少内存占用，"
        "并使用分层缓存将权重分布在 GPU、内存与磁盘之间。"
        "推理引擎采用逐层预读取与专家按需加载策略，"
        "在有限显存下实现了长文本的高效生成。") * 3
ids = tok(text, return_tensors='pt').input_ids[:, :128].to(vs.DEVICE)
print(f'[pf] prompt {ids.shape[1]} tokens, prefill...', flush=True)
t0 = time.time()
logits, cache = eng.forward(ids)
dt = time.time() - t0
print(f'[pf] prefill 128 tok in {dt:.1f}s = {ids.shape[1]/dt:.2f} tok/s', flush=True)
print('[pf] DONE', flush=True)
