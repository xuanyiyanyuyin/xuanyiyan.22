# -*- coding: utf-8 -*-
"""[2026-10-03] 引擎运行包装器：跑目标脚本，无论正常结束/异常/被 SystemExit，结尾一律强制归还内存。

用法（面板/任何地方）：python -u mem_release_runner.py <目标脚本> [参数...]
归还动作：删大对象 → gc×3 → torch cuda empty_cache/ipc_collect → EmptyWorkingSet → 打印前后提交量
"""
import gc, os, runpy, sys, time


def _commit_gb():
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPageFile - m.ullAvailPageFile) / 2**30, m.ullAvailPhys / 2**30


def release(tag='runner'):
    before_c, before_f = _commit_gb()
    # ① 丢掉除自身以外所有大对象引用（模块级缓存/管线）
    keep = set(globals().keys()) | {'keep', 'release', '_commit_gb', 'tag'}
    for name in list(sys.modules.keys()):
        mod = sys.modules.get(name)
        if mod is None or name.startswith(('__', 'sys', 'os', 'gc', 'runpy')):
            continue
        for attr in list(vars(mod).keys()):
            try:
                v = getattr(mod, attr)
            except Exception:
                continue
            if isinstance(v, (dict, list)) and len(v) > 200:
                try:
                    v.clear()
                except Exception:
                    pass
    gc.collect(); gc.collect(); gc.collect()
    # ② torch 侧：显存/固定内存/IPC 缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    # ③ 归还工作集（对文件映射页作用有限，但对堆内存有效）
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1F0FFF, False, os.getpid())
        if h:
            k32.SetProcessWorkingSetSize(h, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
            k32.CloseHandle(h)
    except Exception:
        pass
    gc.collect()
    time.sleep(0.3)
    after_c, after_f = _commit_gb()
    print('[mem-release:%s] 提交 %.1f→%.1f GB（归还 %.1f） | 物理可用 %.1f→%.1f GB（+%.1f）'
          % (tag, before_c, after_c, before_c - after_c, before_f, after_f, after_f - before_f), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python mem_release_runner.py <目标脚本> [参数...]')
        sys.exit(2)
    target = sys.argv[1]
    code_arg = None
    if target == '-c':
        # [2026-10-03] 支持 python -c 形式的引擎分支（面板很多分支是 -c 传代码）
        code_arg = sys.argv[2]
        sys.argv = sys.argv[2:]
        target = 'inline:-c'
    else:
        sys.argv = sys.argv[1:]
    rc = 0
    t0 = time.time()
    try:
        if code_arg is not None:
            exec(compile(code_arg, '<engine -c>', 'exec'), {'__name__': '__main__'})
        else:
            runpy.run_path(target, run_name='__main__')
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except BaseException as e:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        try:
            release(os.path.basename(target))
        except Exception as e:
            print('[mem-release] 归还失败: %s' % str(e)[:120], flush=True)
    print('[mem-release] %s 结束 rc=%s 用时 %.0fs' % (os.path.basename(target), rc, time.time() - t0), flush=True)
    sys.exit(rc)
