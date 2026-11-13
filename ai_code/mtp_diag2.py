# -*- coding: utf-8 -*-
"""MTP 卡死定位: 层/注意力/MLP 逐层计时 + 堆栈采样 + CUDA 探活"""
import sys, os, threading, time, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')

OUT = r'D:/ai/mtp_diag2_out.log'
LOG = open(OUT, 'w', encoding='utf-8', buffering=1)
_lock = threading.Lock()
def log(*a):
    with _lock:
        s = ' '.join(str(x) for x in a)
        LOG.write(s + '\n')
        LOG.flush()
        print(s, flush=True)

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer, DeepseekV4Attention)

# ─── 1. 逐层计时 (DecoderLayer.forward 前后) ───
_orig_layer_fwd = DeepseekV4DecoderLayer.forward
def traced_layer_fwd(self, hidden_states, **kw):
    lid = self.layer_idx
    t0 = time.time()
    log(f'[L]  entry layer {lid} dtype={hidden_states.dtype}')
    try:
        r = _orig_layer_fwd(self, hidden_states, **kw)
        log(f'[L]  exit  layer {lid} ({time.time()-t0:.2f}s)')
        return r
    except Exception as e:
        log(f'[L]  EXC   layer {lid}: {str(e)[:150]}')
        raise
DeepseekV4DecoderLayer.forward = traced_layer_fwd

# ─── 2. 注意力计时 ───
_orig_attn_fwd = DeepseekV4Attention.forward
def traced_attn_fwd(self, hidden_states, **kw):
    t0 = time.time()
    try:
        r = _orig_attn_fwd(self, hidden_states, **kw)
        return r
    except Exception as e:
        log(f'[A]  EXC attn layer {self.layer_idx}: {str(e)[:150]}')
        raise
    finally:
        dt = time.time() - t0
        if dt > 0.5:
            log(f'[A]  attn layer {self.layer_idx} SLOW ({dt:.2f}s)')
DeepseekV4Attention.forward = traced_attn_fwd

# ─── 3. V4LazyMoE 计时 (mlp) ───
_orig_lazy_fwd = None
import v4flash_chunked_sched as vs
_orig_lazy_fwd = vs.V4LazyMoE.forward
def traced_lazy_fwd(self, hidden_states, input_ids=None):
    t0 = time.time()
    try:
        r = _orig_lazy_fwd(self, hidden_states, input_ids)
        return r
    except Exception as e:
        log(f'[M]  EXC mlp l{self.lid} mtp={self.mtp}: {str(e)[:150]}')
        raise
    finally:
        dt = time.time() - t0
        if dt > 0.5:
            log(f'[M]  mlp l{self.lid} mtp={self.mtp} SLOW ({dt:.2f}s)')
vs.V4LazyMoE.forward = traced_lazy_fwd

# ─── 4. 堆栈采样 + CUDA 探活 (每 4 秒) ───
def sampler():
    while True:
        time.sleep(4)
        # 主线程 Python 栈 (纯读取, 不阻塞)
        for t in threading.enumerate():
            if t.name == 'MainThread' and t.ident:
                frames = sys._current_frames()
                f = frames.get(t.ident)
                if f:
                    stack = []
                    while f:
                        stack.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} {f.f_code.co_name}")
                        f = f.f_back
                    log('[S]  main:\n       ' + '\n       '.join(stack[:16]))
                break
threading.Thread(target=sampler, daemon=True).start()

# ─── 5. 跑 MTP 测试 ───
log('=== 开始加载引擎 ===')
eng = vs.V4FlashEngine()
log(f'=== 引擎加载完成, MTP={"有" if eng.mtp is not None else "无"} ===')

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好', return_tensors='pt').input_ids.to(vs.DEVICE)
log('=== 开始 generate (max_new_tokens=8) ===')
t_start = time.time()
try:
    out = eng.generate(ids, max_new_tokens=8, do_sample=False)
    log(f'=== 生成完成 ({time.time()-t_start:.0f}s): {repr(tok.decode(out[0], skip_special_tokens=True))} ===')
except Exception as e:
    log(f'=== 生成异常: {str(e)[:200]} ===')
    traceback.print_exc(file=LOG)
    LOG.flush()
log('=== DONE ===')
