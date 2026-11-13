# -*- coding: utf-8 -*-
"""_read 内部拆解 — f.read(磁盘IO) vs frombuffer+clone(Python/torch层)
引擎内 decode 时统计, 定位 pread 慢根因:
  如果 f.read 慢 → 磁盘IO/排队  |  如果 clone 慢 → GIL/内存带宽/Python层
"""
import sys, time, struct, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# 计时段
SEG = {'fread_s': 0.0, 'fread_n': 0, 'clone_s': 0.0, 'clone_n': 0,
       'fread_max': 0.0, 'clone_max': 0.0}

_orig_read = vs.ExpertReader._read
def _timed_read(self, key):
    shard = self.disk[key]
    f = self._fh(shard)
    hd = self._hdr_cache.get(shard)
    if hd is None:
        with open(self._shard_path(shard), 'rb') as tf:
            n = struct.unpack('<Q', tf.read(8))[0]
            hd = (n + 8, json.loads(tf.read(n)))
        self._hdr_cache[shard] = hd
    base, hdr = hd
    m = hdr[key]
    o0, o1 = m['data_offsets']
    t0 = time.time()
    f.seek(base + o0)
    data = f.read(o1 - o0)
    dt_f = time.time() - t0
    SEG['fread_s'] += dt_f; SEG['fread_n'] += 1
    if dt_f > SEG['fread_max']: SEG['fread_max'] = dt_f
    t0 = time.time()
    r = torch.frombuffer(data, dtype=vs.DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone()
    dt_c = time.time() - t0
    SEG['clone_s'] += dt_c; SEG['clone_n'] += 1
    if dt_c > SEG['clone_max']: SEG['clone_max'] = dt_c
    return r
vs.ExpertReader._read = _timed_read

print('[ri] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
ids = tok("你好，介绍一下", return_tensors='pt').input_ids.to(vs.DEVICE)
logits, cache, h = eng.forward(ids, return_hidden=True)

for n in range(3):
    A = logits[:, -1, :].argmax(-1, keepdim=True)
    logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)

fn = SEG['fread_n']; cn = SEG['clone_n']
print(f'[ri] === 引擎内 _read 拆解 (decode 3 token) ===', flush=True)
if fn:
    print(f'[ri] f.read(磁盘IO):  n={fn} avg={SEG["fread_s"]/fn*1000:.1f}ms max={SEG["fread_max"]*1000:.0f}ms total={SEG["fread_s"]:.2f}s', flush=True)
if cn:
    print(f'[ri] clone(torch):   n={cn} avg={SEG["clone_s"]/cn*1000:.1f}ms max={SEG["clone_max"]*1000:.0f}ms total={SEG["clone_s"]:.2f}s', flush=True)
# 引擎外对照
SEG['fread_s']=SEG['fread_n']=SEG['clone_s']=SEG['clone_n']=SEG['fread_max']=SEG['clone_max']=0
reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)
for i in range(20):
    reader._read(f'layers.{i%43}.ffn.experts.{i}.w1.weight')
fn = SEG['fread_n']; cn = SEG['clone_n']
print(f'[ri] === 引擎外空闲 _read 拆解 (20次) ===', flush=True)
if fn:
    print(f'[ri] f.read(磁盘IO):  avg={SEG["fread_s"]/fn*1000:.1f}ms max={SEG["fread_max"]*1000:.0f}ms', flush=True)
if cn:
    print(f'[ri] clone(torch):   avg={SEG["clone_s"]/cn*1000:.1f}ms max={SEG["clone_max"]*1000:.0f}ms', flush=True)
print('[ri] DONE', flush=True)
