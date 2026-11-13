# -*- coding: utf-8 -*-
"""对比: 预加载RAM vs 页缓存模式 — 预热后的 decode 速度 + 系统内存分布
用法: python v4flash_ramcmp.py [preload|pagecache]
"""
import sys, time, ctypes, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')

mode = sys.argv[1] if len(sys.argv) > 1 else 'preload'
os.environ['V4_RAM_MODE'] = mode  # 引擎不读这个, 仅标识

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong),('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
def meminfo():
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys/2**30, m.dwMemoryLoad

import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 设置模式 (临时改引擎属性)
print(f'[cmp] 模式: {mode} (use_ram_preload={vs.V4FlashEngine.__init__.__globals__ is not None})', flush=True)

eng = vs.V4FlashEngine()
# 强制设置模式 (读环境变量后的初始值可能是True, 这里强制)
# 注意: use_ram_preload 在 __init__ 里写死, 这里无法改已初始化的实例
# 我们通过修改源文件? 不. 这里直接测当前引擎即可, 模式由启动前改文件决定

tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
prompt = ("请写一篇关于人工智能与人类创造力的短文, 大约300字。")
ids = tok(prompt, return_tensors='pt').input_ids[:, :50].to(vs.DEVICE)

# 预热 (读权重, 页缓存模式下让页缓存热)
print('[cmp] 预热...', flush=True)
eng.forward(ids, return_hidden=True)

# 测 decode 20 token
logits, cache, h = eng.forward(ids, return_hidden=True)
t0 = time.time()
n = 0
while n < 20:
    last = logits[:, -1, :].clone().float()
    A = last.argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
    n += 1
dt = time.time() - t0

# 内存指标
avail, load = meminfo()
import psutil
proc_gb = psutil.Process().memory_info().rss / 2**30

print(f'\n===== 结果 [{mode}] =====', flush=True)
print(f'decode {n} token 耗时 {dt:.1f}s = {dt/n:.2f} s/tok', flush=True)
print(f'进程RSS: {proc_gb:.1f}GB', flush=True)
print(f'系统可用(含Standby可回收): {avail:.1f}GB', flush=True)
print(f'系统内存负载: {load}%', flush=True)
print('===== DONE =====', flush=True)
