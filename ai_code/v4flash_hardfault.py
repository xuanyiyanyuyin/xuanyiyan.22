# -*- coding: utf-8 -*-
"""硬缺页测量 — decode 期间系统 Pages Input/sec (真正读页文件/磁盘的硬缺页)
区分软缺页(standby恢复,便宜) vs 硬缺页(页文件IO,贵) — 判断内存换页是否真瓶颈
"""
import sys, time, subprocess, threading
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

def ps_counter(counter, seconds, interval=1):
    """PowerShell 采样性能计数器, 返回值列表"""
    cmd = ['powershell', '-NoProfile', '-Command',
           f"Get-Counter '{counter}' -SampleInterval {interval} -MaxSamples {seconds} " +
           "| ForEach-Object { $_.CounterSamples.CookedValue }"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds+10)
        vals = [float(x) for x in r.stdout.strip().split() if x.replace('.','').replace('-','').isdigit()]
        return vals
    except Exception as e:
        return [f'ERR {e}'.strip()]

print('[hf] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok('你好，介绍一下', return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)
A = logits[:, -1, :].argmax(-1, keepdim=True)
logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

# 后台 decode 线程
stop = threading.Event()
def decode_loop():
    _l, _c, _h = logits, cache, h
    while not stop.is_set():
        A = _l[:, -1, :].argmax(-1, keepdim=True)
        _l, _c, _h = eng.forward(A, past_key_values=_c, return_hidden=True)
d = threading.Thread(target=decode_loop, daemon=True)

print('[hf] 空闲 3s 采样基线...', flush=True)
v_idle_pg = ps_counter(r'\Memory\Pages Input/sec', 3)
v_idle_av = ps_counter(r'\Memory\Available MBytes', 3)
print(f'[hf] 空闲: Pages Input/sec={v_idle_pg} | Available={v_idle_av}MB', flush=True)

print('[hf] decode 中 5s 采样...', flush=True)
d.start()
time.sleep(1)  # 让 decode 稳定
v_pg = ps_counter(r'\Memory\Pages Input/sec', 4)
v_av = ps_counter(r'\Memory\Available MBytes', 4)
v_ws = ps_counter(r'\Process(python*)\Working Set', 4)
stop.set(); d.join()
print(f'[hf] decode: Pages Input/sec={v_pg}', flush=True)
print(f'[hf] decode: Available={v_av}MB', flush=True)
print(f'[hf] decode: python WorkingSet={v_ws}MB', flush=True)
# Pages Input/sec 是"每秒读入页数", 乘页大小(4KB)得每秒读入字节
if v_pg and all(isinstance(x, float) for x in v_pg):
    avg = sum(v_pg)/len(v_pg)
    print(f'[hf] 硬缺页速率: {avg:.0f} pages/s = {avg*4096/1e6:.1f} MB/s (页文件+映射文件读)', flush=True)
print('[hf] DONE', flush=True)
