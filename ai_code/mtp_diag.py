import faulthandler, sys, os, threading
faulthandler.dump_traceback_later(90, exit=True)  # 90秒后 dump 线程栈并退出
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
import torch
from transformers import AutoTokenizer

print('[diag] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
print('[diag] 引擎加载完成, MTP:', '有' if eng.mtp is not None else '无', flush=True)

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好', return_tensors='pt').input_ids.to(vs.DEVICE)
print('[diag] 开始 generate (max_new_tokens=8)...', flush=True)
out = eng.generate(ids, max_new_tokens=8, do_sample=False)
print('[diag] 生成完成:', repr(tok.decode(out[0], skip_special_tokens=True)), flush=True)
