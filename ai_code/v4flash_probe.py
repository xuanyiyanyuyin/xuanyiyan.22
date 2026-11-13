# -*- coding: utf-8 -*-
"""V4Flash 长decode 详细探针 — 定位变慢真凶
后台每2s采样系统指标 + 主线程每10token采样, 对齐"变慢前一刻谁在涨"
指标: 进程RSS / 可用内存 / Memory Compression / 系统缓存(页缓存) / 磁盘读速 / CPU / GPU
用法: D:\ai\python_ltx25\Scripts\python.exe -u D:\ai\v4flash_probe.py
建议: 跑前停掉其他占内存进程(游戏/qianwen等), 确保 V4Flash 独享, 才能定位自身问题
"""
import sys, time, ctypes, threading, subprocess, os, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import psutil
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

# ═══ 模型内部计时 (monkey-patch, 累积每token分解) ═══
INT = collections.defaultdict(float)   # 每token内各环节耗时
INT_CNT = collections.defaultdict(int)
_tok_start = [0.0]

# V4LazyMoE: router / 专家读取等待 / GEMM
_orig_lazy = vs.V4LazyMoE.forward
def _timed_lazy(self, hidden_states, input_ids=None):
    B, S, H = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.reshape(-1, H)
    t0 = time.time()
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    INT['router'] += time.time() - t0
    uids = indices.unique().tolist()
    INT_CNT['expert'] += len(uids)
    if self.mtp:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_mtp_expert, e) for e in uids}
    else:
        futs = {e: self.engine.expert_pool.submit(self.engine.get_expert, self.lid, e) for e in uids}
    out = torch.zeros(flat.shape[0], H, device=flat.device, dtype=torch.float32)
    t_r = time.time()
    results = {e: futs[e].result() for e in uids}
    INT['expert_wait'] += time.time() - t_r
    t_g = time.time()
    for eid in uids:
        w1, s1, w2, s2, w3, s3 = results[eid]
        hit = (indices == eid)
        tidx = hit.any(-1).nonzero().squeeze(-1)
        if tidx.numel() == 0:
            continue
        x = flat[tidx]
        dev = x.device
        g = vs.FP8_KERNEL.matmul_2d(x, w1.to(dev), s1.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        u = vs.FP8_KERNEL.matmul_2d(x, w3.to(dev), s3.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        gate = g.clamp(max=self.limit); up = u.clamp(min=-self.limit, max=self.limit)
        inter = torch.nn.functional.silu(gate) * up
        d = vs.FP8_KERNEL.matmul_2d(inter, w2.to(dev), s2.to(dev), [vs.BLOCK_SIZE, vs.BLOCK_SIZE], torch.bfloat16).float()
        w = (weights * hit).sum(-1)[tidx]
        out[tidx] += d * w.unsqueeze(-1)
    INT['gemm'] += time.time() - t_g
    shared = self.shared_experts(residual).to(torch.float32)
    return (out.view(B, S, H) + shared).to(hidden_states.dtype)
vs.V4LazyMoE.forward = _timed_lazy

# MTP 模块
_orig_mtp = vs.V4MTPModule.forward
def _timed_mtp(self, token_ids, hidden_states, position_ids, cache):
    t0 = time.time()
    r = _orig_mtp(self, token_ids, hidden_states, position_ids, cache)
    INT['mtp'] += time.time() - t0
    return r
vs.V4MTPModule.forward = _timed_mtp

# 层构建 (build_and_inject)
_orig_build = vs.V4FlashEngine.build_and_inject
def _timed_build(self, lid):
    t0 = time.time()
    r = _orig_build(self, lid)
    INT['layer_build'] += time.time() - t0
    return r
vs.V4FlashEngine.build_and_inject = _timed_build

# 主 forward
_orig_fwd = vs.V4FlashEngine.forward
def _timed_fwd(self, input_ids, past_key_values=None, return_hidden=False):
    t0 = time.time()
    r = _orig_fwd(self, input_ids, past_key_values, return_hidden)
    INT['fwd_total'] += time.time() - t0
    INT_CNT['fwd'] += 1
    return r
vs.V4FlashEngine.forward = _timed_fwd

# ═══ Windows 内存 API ═══
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong),('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
def mem_status():
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m

class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong),('CommitTotal', ctypes.c_size_t),('CommitLimit', ctypes.c_size_t),
                ('CommitPeak', ctypes.c_size_t),('PhysicalTotal', ctypes.c_size_t),('PhysicalAvailable', ctypes.c_size_t),
                ('SystemCache', ctypes.c_size_t),('KernelTotal', ctypes.c_size_t),('KernelPaged', ctypes.c_size_t),
                ('KernelNonPaged', ctypes.c_size_t),('PageSize', ctypes.c_size_t),('HandleCount', ctypes.c_ulong),
                ('ProcessCount', ctypes.c_ulong),('ThreadCount', ctypes.c_ulong)]
def perf_info():
    p = PERFORMANCE_INFORMATION(); p.cb = ctypes.sizeof(PERFORMANCE_INFORMATION)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p

def mem_compression_gb():
    """Memory Compression 进程占用 (内存压力时膨胀)"""
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            if p.info['name'] and 'MemCompression' in p.info['name']:
                return p.info['memory_info'].rss / 2**30
        except Exception:
            pass
    return 0.0

# ═══ 系统采样线程 (事件触发, 每token同步采样) ═══
sys_samples = []
stop_sampling = False
sample_event = threading.Event()
def _collect_one():
    m = mem_status()
    pi = perf_info()
    mc = mem_compression_gb()
    disk = psutil.disk_io_counters()
    cpu = psutil.cpu_percent(interval=None)
    gpu = ''
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        gpu = r.stdout.strip()
    except Exception:
        pass
    return {
        't': time.time(), 'avail': m.ullAvailPhys / 2**30, 'load': m.dwMemoryLoad,
        'commit_avail': m.ullAvailPageFile / 2**30,
        'syscache': pi.SystemCache * pi.PageSize / 2**30,
        'memcomp': mc, 'disk_MB_s': 0, 'cpu%': cpu, 'gpu': gpu,
    }

_last_disk = [psutil.disk_io_counters()]
_last_disk_t = [time.time()]
def _disk_rate():
    disk = psutil.disk_io_counters()
    now = time.time()
    dt = now - _last_disk_t[0]
    rate = (disk.read_bytes - _last_disk[0].read_bytes) / dt / 2**20 if dt > 0 else 0
    _last_disk[0], _last_disk_t[0] = disk, now
    return rate

def sampler():
    while not stop_sampling:
        sample_event.wait()      # 等主线程每token触发
        sample_event.clear()
        try:
            s = _collect_one()
            s['disk_MB_s'] = _disk_rate()
            sys_samples.append(s)
        except Exception as e:
            sys_samples.append({'t': time.time(), 'err': str(e)[:50]})

# ═══ 主流程 ═══
def main():
    print('[probe] 加载引擎...', flush=True)
    eng = vs.V4FlashEngine()
    tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
    prompt = ("请写一篇关于人工智能与人类创造力的深度长文, "
              "从历史、现状、未来三个角度展开, 包含具体例子和你的思考, 尽量写长一些。")
    ids = tok(prompt, return_tensors='pt').input_ids[:, :50].to(vs.DEVICE)
    print(f'[probe] prompt {ids.shape[1]} token', flush=True)

    # 预热
    print('[probe] 预热...', flush=True)
    eng.forward(ids, return_hidden=True)

    threading.Thread(target=sampler, daemon=True).start()

    logits, cache, h = eng.forward(ids, return_hidden=True)
    proc = psutil.Process()
    t_start = time.time()
    n = 0
    print('\n[probe] 长decode开始 (目标100 token, 每token采样, 系统+模型内部)', flush=True)
    print('tok | 本轮s | 进程GB | 可用GB | MemC | 系统缓存 | 磁盘MB/s | CPU% | fwd内部分解(router/专家等待/gemm/层构建/mtp/其他)', flush=True)
    while n < 100:
        last = logits[:, -1, :].clone().float()
        A = last.argmax(-1, keepdim=True)
        t_round = time.time()
        for k in INT: INT[k] = 0.0     # 清零内部计时, 只记本轮
        logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
        n += 1
        sample_event.set()
        s = sys_samples[-1] if sys_samples else {}
        dt = time.time() - t_round
        rss = proc.memory_info().rss / 2**30
        # 内部分解: 本轮 fwd_total 内各环节占比
        ft = INT.get('fwd_total', 0)
        r = INT.get('router', 0); ew = INT.get('expert_wait', 0); gm = INT.get('gemm', 0)
        lb = INT.get('layer_build', 0); mp = INT.get('mtp', 0)
        other = max(0, ft - r - ew - gm - lb - mp)
        print(f'{n:3d} | {dt:5.2f} | {rss:4.1f} | {s.get("avail",0):4.1f} | {s.get("memcomp",0):4.2f} | '
              f'{s.get("syscache",0):5.1f} | {s.get("disk_MB_s",0):6.0f} | {s.get("cpu%",0):3.0f} | '
              f'rt{int(r*1000)}/ew{int(ew*1000)}/gm{int(gm*1000)}/lb{int(lb*1000)}/mtp{int(mp*1000)}/ot{int(other*1000)}ms', flush=True)
        # 每30 token 强制一次 gc, 观察是否内存泄漏
        if n % 30 == 0:
            import gc; gc.collect()

    print('\n[probe] 系统采样历史 (最后25条):', flush=True)
    for s in sys_samples[-25:]:
        print(f'  t={s.get("t",0):.0f} 可用={s.get("avail",0):.1f}GB MemC={s.get("memcomp",0):.2f}GB '
              f'系统缓存={s.get("syscache",0):.1f}GB 磁盘={s.get("disk_MB_s",0):.0f}MB/s CPU={s.get("cpu%",0):.0f}% '
              f'commit可用={s.get("commit_avail",0):.0f}GB', flush=True)
    print('\n[probe] 说明:', flush=True)
    print('  - 若 进程GB 持续涨 → 内存泄漏/缓存膨胀', flush=True)
    print('  - 若 MemC GB 涨 → Windows 内存压缩触发(内存压力)', flush=True)
    print('  - 若 系统缓存GB 满但可用低 → 页缓存挤占', flush=True)
    print('  - 若 磁盘MB/s 高但慢 → 磁盘IO瓶颈', flush=True)
    print('  - 若 CPU% 被抢(其他进程) → 并发竞争', flush=True)
    print('[probe] DONE', flush=True)

if __name__ == '__main__':
    main()
