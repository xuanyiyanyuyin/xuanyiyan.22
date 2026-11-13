# -*- coding: utf-8 -*-
"""快速验证: mtp_steps=1 级联代码是否正常 (max_new_tokens=2)"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
import torch
from transformers import AutoTokenizer

print('[quick] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
print(f'[quick] 引擎加载完成 MTP={"有" if eng.mtp is not None else "无"} mtp_steps={eng.mtp_steps}', flush=True)

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好', return_tensors='pt').input_ids.to(vs.DEVICE)
print('[quick] 开始 generate (max_new_tokens=2)...', flush=True)
t1 = time.time()
out = eng.generate(ids, max_new_tokens=2, do_sample=False)
print(f'[quick] 生成完成 ({time.time()-t1:.0f}s): {repr(tok.decode(out[0], skip_special_tokens=True))}', flush=True)
print('[quick] DONE', flush=True)
