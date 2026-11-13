#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flux2_encode_gen.py —— FLUX.2「先编码 → 再生图」一键流程（面板 f2auto 卡走这个）

用法:
  python flux2_encode_gen.py "<prompt>" [steps=4] [width=512] [height=512] [resident_d=0] [参考图,逗号分隔]

流程（关键：两个阶段必须**分进程**，45G text_encoder 与 66G transformer 不能同场）:
  ① 按引擎规则算缓存名 `D:\\ai\\_cached_flux2_embeds_<md5(prompt|512)[:16]>.pt`
  ② 命中就跳过；没有（或 F2_FORCE_ENCODE=1）就起 **子进程** 跑 flux2_encode.py 生成缓存
  ③ 子进程退出后，本进程再 import flux2_3tier 走 text2img_f2(..., embeds_file=<缓存>)
  ④ 末行打印 RESULT:<图片路径>（引擎管理器按这个协议解析）

环境变量:
  F2_FORCE_ENCODE=1  → 无视已有缓存，强制重新编码（提示词没变但想重编时用）
"""
import hashlib
import os
import subprocess
import sys
import time

PY = r'D:\ai\python\python.exe'
ENC = r'D:\ai\flux2_encode.py'
MAX_SEQ = 512
FMT_TAG = 'fmt2'   # ★ 必须与 flux2_encode.py 一致：官方 system_message + 取层 (i+1)


def cache_path_for(prompt):
    key = hashlib.md5(f'{prompt}|{MAX_SEQ}|{FMT_TAG}'.encode()).hexdigest()[:16]
    return rf'D:\ai\_cached_flux2_embeds_{key}.pt'


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'a cute orange cat'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    rd = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    refs = [p.strip() for p in (sys.argv[6].split(',') if len(sys.argv) > 6 else []) if p.strip()]
    steps = max(1, min(int(steps), 100))   # [2026-10-03] 上限 60→100（50 以上无增益，留实验余量）
    force = os.environ.get('F2_FORCE_ENCODE', '0') == '1'
    cache = cache_path_for(prompt)

    t0 = time.perf_counter()
    if force or not os.path.exists(cache):
        print(f'[f2auto] 无缓存 → 先编码（独立进程）: {cache}', flush=True)
        p = subprocess.run([PY, '-u', ENC, prompt, str(MAX_SEQ), cache],
                           cwd=r'D:\ai', capture_output=True, text=True, timeout=7200)
        tail = '\n'.join((p.stdout or '').splitlines()[-12:])
        print(tail, flush=True)
        if p.returncode != 0 or not os.path.exists(cache):
            print('[f2auto] 编码失败：' + (p.stderr or '')[-1200:], flush=True)
            sys.exit(2)
        print(f'[f2auto] 编码完成 {time.perf_counter() - t0:.0f}s', flush=True)
    else:
        print(f'[f2auto] 命中缓存，跳过编码: {cache}', flush=True)

    sys.path.insert(0, r'D:\ai')
    # ══ [2026-10-02] 内存卫生：mmap 进来的权重页是干净的，用完/退出时归还给系统 ══
    #    自带 sys.path 兜底 —— 放哪都不会 ImportError（2026-10-02 踩过：块被插到
    #    sys.path.insert 之前，导致 flux2_encode_gen 一加载就 ModuleNotFoundError）
    try:
        import sys as _mh_sys
        if r'D:\ai' not in _mh_sys.path:
            _mh_sys.path.insert(0, r'D:\ai')
        import mem_hygiene as _mh
        _mh.trim('start')
        import atexit as _atexit
        _atexit.register(lambda: _mh.trim('exit'))
    except Exception:
        _mh = None

    import flux2_3tier as f2  # 延迟 import：确保编码子进程已退出、内存已释放

    print(f'[f2auto] 开始生图 steps={steps} {width}x{height} rd={rd} refs={len(refs)}', flush=True)
    fn = f2.text2img_f2(prompt, steps=steps, width=width, height=height,
                        resident_d=rd, images=(refs or None), embeds_file=cache)
    print(f'[f2auto] 出图 {time.perf_counter() - t0:.0f}s', flush=True)
    print('RESULT:' + fn, flush=True)


if __name__ == '__main__':
    main()
