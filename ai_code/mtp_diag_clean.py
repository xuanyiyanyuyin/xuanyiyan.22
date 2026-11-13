# -*- coding: utf-8 -*-
"""干净复测: 无埋点/无 faulthandler, 验证 MTP 在 build_pool=1/expert_pool=12 下不再卡死"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
import torch
from transformers import AutoTokenizer

print('[clean] 加载引擎...', flush=True)
t0 = time.time()
eng = vs.V4FlashEngine()
print(f'[clean] 引擎加载完成 MTP={"有" if eng.mtp is not None else "无"} ({time.time()-t0:.0f}s)', flush=True)

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好', return_tensors='pt').input_ids.to(vs.DEVICE)
print('[clean] 开始 generate (max_new_tokens=8)...', flush=True)
t1 = time.time()
out = eng.generate(ids, max_new_tokens=8, do_sample=False)
print(f'[clean] 生成完成 ({time.time()-t1:.0f}s): {repr(tok.decode(out[0], skip_special_tokens=True))}', flush=True)
print('[clean] DONE', flush=True)
