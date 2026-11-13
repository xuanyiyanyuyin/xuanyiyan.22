# -*- coding: utf-8 -*-
"""V4Flash 带宽探针 — 每token测量 PCIe(H2D/D2H) / 显存利用率 / GPU算力 / 磁盘带宽
后台 nvidia-smi dmon 采集 sm/mem/rxpci/txpci, 主线程每token对齐
"""
import sys, time, threading, subprocess, psutil
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import torch
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

N_TOK = 20
bw_samples = []
_stop = False
def dmon_reader():
    try:
        p = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'ut', '-d', '1'],
                             stdout=subprocess.PIPE, text=True, stderr=subprocess.DEVNULL)
        for line in p.stdout:
            if _stop:
                break
            parts = line.split()
            if len(parts) >= 9 and parts[0].isdigit():
                bw_samples.append({'t': time.time(), 'sm': int(parts[1]), 'mem': int(parts[2]),
                                   'rx': float(parts[7]), 'tx': float(parts[8])})
        p.kill()
    except Exception:
        pass

_last_disk = [psutil.disk_io_counters()]
_last_disk_t = [time.time()]
def disk_rate():
    d = psutil.disk_io_counters()
    now = time.time()
    dt = now - _last_disk_t[0]
    rate = (d.read_bytes - _last_disk[0].read_bytes) / dt / 2**20 if dt > 0 else 0
    _last_disk[0], _last_disk_t[0] = d, now
    return rate

def main():
    print('[bw] 加载引擎...', flush=True)
    eng = vs.V4FlashEngine()
    tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
    prompt = "请写一篇关于人工智能与人类创造力的短文, 大约200字。"
    ids = tok(prompt, return_tensors='pt').input_ids[:, :40].to(vs.DEVICE)
    print('[bw] 预热...', flush=True)
    eng.forward(ids, return_hidden=True)

    threading.Thread(target=dmon_reader, daemon=True).start()
    time.sleep(2)  # dmon 启动

    logits, cache, h = eng.forward(ids, return_hidden=True)
    print('\n[bw] decode 开始 (20 token)', flush=True)
    print('tok | 本轮s | sm% | mem% | PCIe RX(H2D)MB/s | PCIe TX(D2H)MB/s | 磁盘MB/s', flush=True)
    rows = []
    for n in range(N_TOK):
        last = logits[:, -1, :].clone().float()
        A = last.argmax(-1, keepdim=True)
        t0 = time.time()
        logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
        dt = time.time() - t0
        s = bw_samples[-1] if bw_samples else {}
        dr = disk_rate()
        rows.append((dt, s.get('sm', 0), s.get('mem', 0), s.get('rx', 0), s.get('tx', 0), dr))
        print(f'{n+1:3d} | {dt:5.2f} | {s.get("sm",0):3d} | {s.get("mem",0):3d} | {s.get("rx",0):6.0f} | {s.get("tx",0):6.0f} | {dr:6.0f}', flush=True)

    # 汇总
    if rows:
        avg_dt = sum(r[0] for r in rows) / len(rows)
        avg_sm = sum(r[1] for r in rows) / len(rows)
        avg_mem = sum(r[2] for r in rows) / len(rows)
        avg_rx = sum(r[3] for r in rows) / len(rows)
        avg_tx = sum(r[4] for r in rows) / len(rows)
        avg_disk = sum(r[5] for r in rows) / len(rows)
        print('\n===== 带宽汇总 (平均/轮) =====', flush=True)
        print(f'  每轮 {avg_dt:.2f}s = {1/avg_dt:.2f} tok/s', flush=True)
        print(f'  GPU算力 sm%: {avg_sm:.0f} (100%=算力饱和)', flush=True)
        print(f'  显存利用率 mem%: {avg_mem:.0f} (100%=显存带宽饱和)', flush=True)
        print(f'  PCIe RX (H2D): {avg_rx:.0f} MB/s (PCIe4.0x8理论~14000)', flush=True)
        print(f'  PCIe TX (D2H): {avg_tx:.0f} MB/s', flush=True)
        print(f'  磁盘读: {avg_disk:.0f} MB/s (NVMe理论~3000)', flush=True)
        print('\n  ⚠️ 判断:')
        print(f'    mem% 高({avg_mem:.0f}%) → 显存带宽瓶颈', flush=True)
        print(f'    PCIe RX 接近带宽 → H2D 金手指瓶颈', flush=True)
        print(f'    磁盘高 → 专家读盘瓶颈', flush=True)
        print(f'    sm% 高 → GPU 算力瓶颈', flush=True)
    _stop = True
    print('\n[bw] DONE', flush=True)

if __name__ == '__main__':
    main()
