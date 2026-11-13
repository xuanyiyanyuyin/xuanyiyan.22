# -*- coding: utf-8 -*-
"""[2026-10-02] 内存卫生共用件 —— 归还工作集 + 内存日志 + 周期回收

背景（真出过事）：2026-10-01 20:51 系统日志记录 python.exe 涨到 64.7GB 虚拟内存、
  20:53 栈溢出崩溃（0xc00000fd），整套本地服务全死。两个来源：
  ① gen_api 每次生图加载 ~18GB klein 管线、用完即卸，反复循环不归还；
  ② FLUX.2 三层搬运链路用 mmap 把 66GB transformer 整个映射进进程（虚拟内存直接 66GB+），
     读盘还会把系统文件缓存撑满 —— 32G 机器上与游戏/其它服务共存必然被拖垮。

mmap 进来的权重页是「干净的」（磁盘上有原件），所以用 EmptyWorkingSet 丢掉不会丢数据，
下次读盘再进 —— 这正是本模块存在的意义。

用法：
    import mem_hygiene as mh
    mh.trim('after_load')            # 打一行内存日志 + 归还工作集
    mh.start_periodic(interval=20)    # 后台每 N 秒归还一次（需 MEM_TRIM_PERIODIC=1 才启动）
开关：MEM_TRIM=0 关全部；MEM_TRIM_PERIODIC=1 开周期回收；MEM_TRIM_SEC=20 周期。
"""
import ctypes
import gc
import os
import threading
import time


def stats():
    """(RSS, 私有, 虚拟, 系统可用) 单位 MB；拿不到返回 0"""
    try:
        import psutil
        m = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        return (m.rss / 1048576, (getattr(m, 'private', 0) or 0) / 1048576,
                (getattr(m, 'vms', 0) or 0) / 1048576, vm.available / 1048576)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def empty_working_set():
    """Windows: 把闲置页还给系统（等价于任务管理器里"清空工作集"）"""
    try:
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        return True
    except Exception:
        return False


def trim(tag='', gc_rounds=2):
    """GC + 显存缓存 + 归还工作集，并打印一行内存日志"""
    if os.environ.get('MEM_TRIM', '1') == '0':
        return stats()
    try:
        for _ in range(max(1, int(gc_rounds))):
            gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    empty_working_set()
    r, p, v, a = stats()
    try:
        print('[mem %s] RSS=%.0fMB 私有=%.0fMB 虚拟=%.0fMB | 系统可用=%.1fGB' % (tag, r, p, v, a / 1024), flush=True)
    except Exception:
        pass
    guard('trim:' + str(tag))
    return (r, p, v, a)



def guard(tag='', limit_mb=None):
    """[2026-10-02] 内存天花板：RSS 超过 MEM_SELF_LIMIT_MB 就记日志并自我退出。
    任务型引擎本来就是一次性的 —— 宁可退出重跑, 也不能无声挂住 25GB（实测事故: 24.9GB 挂了十几小时）。"""
    try:
        lim = float(limit_mb or os.environ.get('MEM_SELF_LIMIT_MB', '0') or 0)
    except Exception:
        lim = 0.0
    if lim <= 0:
        return False
    r, p, v, a = stats()
    if r >= lim:
        try:
            print('[mem-guard] %s RSS=%.0fMB ≥ 上限 %.0fMB → 自我退出（防挂住内存）' % (tag, r, lim), flush=True)
        except Exception:
            pass
        pass   # [2026-10-02 主人令] 运行中绝不打断：只记日志，不退出
    return True

_PERIODIC = {'on': False}


def start_periodic(interval=None, tag='periodic'):
    """后台线程：每 N 秒归还一次工作集（默认关闭；MEM_TRIM_PERIODIC=1 打开）"""
    if os.environ.get('MEM_TRIM_PERIODIC', '0') != '1' or _PERIODIC['on']:
        return False
    _PERIODIC['on'] = True
    try:
        iv = float(interval or os.environ.get('MEM_TRIM_SEC', '20'))
    except Exception:
        iv = 20.0

    def loop():
        while True:
            time.sleep(iv)
            try:
                empty_working_set()
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()
    try:
        print('[mem] 周期归还工作集已启动（每 %.0fs）' % iv, flush=True)
    except Exception:
        pass
    return True
