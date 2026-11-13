# -*- coding: utf-8 -*-
"""合并读验证 v2 — read_expert 6次pread → 2次大块读 (权重块 w1w2w3 + scale块 s1s2s3)
干净实现: 复用 ExpertReader._hdr_cache + _fh, 检查连续性后一次大读切片
验证: ①数值一致 ②耗时对比
"""
import sys, time, json, struct, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs

reader = vs.ExpertReader(vs.MODEL_DIR, vs.DISK)

def _read_block(keys):
    """keys 对应 tensor 若同一 shard 连续, 一次大读切片; 否则逐个读"""
    locs = []
    for k in keys:
        shard = reader.disk[k]
        hd = reader._hdr_cache.get(shard)
        if hd is None:
            with open(reader._shard_path(shard), 'rb') as tf:
                n = struct.unpack('<Q', tf.read(8))[0]
                hd = (n + 8, json.loads(tf.read(n)))
            reader._hdr_cache[shard] = hd
        base, hdr = hd
        m = hdr[k]
        o0, o1 = m['data_offsets']
        locs.append((k, shard, base + o0, o1 - o0, m))
    same_shard = all(l[1] == locs[0][1] for l in locs)
    continuous = same_shard and all(locs[i][2] + locs[i][3] == locs[i+1][2] for i in range(len(locs)-1))
    if continuous:
        f = reader._fh(locs[0][1])
        f.seek(locs[0][2])
        data = f.read(locs[-1][2] + locs[-1][3] - locs[0][2])
        out = []
        for k, shard, off, ln, m in locs:
            s_off = off - locs[0][2]
            out.append(torch.frombuffer(data[s_off:s_off+ln], dtype=vs.DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone())
        return out
    else:
        return [reader._read(k) for k in keys]

def read_merged(lid, eid):
    p = f'layers.{lid}.ffn.experts.{eid}'
    w = _read_block([f'{p}.w1.weight', f'{p}.w2.weight', f'{p}.w3.weight'])
    s = _read_block([f'{p}.w1.scale', f'{p}.w2.scale', f'{p}.w3.scale'])
    return w[0], s[0], w[1], s[1], w[2], s[2]

# ① 正确性: 10 个随机专家
print('[v] 正确性验证 (10 随机专家):', flush=True)
random.seed(3)
ok = True
for i in range(10):
    lid, eid = random.randint(0, 42), random.randint(0, 255)
    ref = reader.read_expert(lid, eid)
    mer = read_merged(lid, eid)
    md = 0
    for j in range(6):
        d = (mer[j].float() - ref[j].float()).abs().max().item()
        md = max(md, d)
    if md != 0:
        ok = False
        print(f'  ({lid},{eid}): MISMATCH diff={md}', flush=True)
    else:
        print(f'  ({lid},{eid}): OK', flush=True)
print(f'[v] {"全部数值一致!" if ok else "存在不一致!"}', flush=True)

# ② 耗时: 单独6读 vs 合并2读 (30 专家, 单线程)
N2 = 30
lids = random.choices(range(43), k=N2)
eids = random.choices(range(256), k=N2)
t0 = time.time()
for i in range(N2): reader.read_expert(lids[i], eids[i])
t_sep = time.time() - t0
t0 = time.time()
for i in range(N2): read_merged(lids[i], eids[i])
t_mer = time.time() - t0
print(f'[v] 单独6读: {t_sep/N2*1000:.1f}ms/专家 | 合并2读: {t_mer/N2*1000:.1f}ms/专家 | 提速 {(t_sep-t_mer)/t_sep*100:.0f}%', flush=True)
print('[v] DONE', flush=True)
