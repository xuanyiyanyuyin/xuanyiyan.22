# -*- coding: utf-8 -*-
r"""玩大型游戏前跑一下: 看这台机器的「可用提交内存」还剩多少
用法:  python D:\ai\check_mem.py
"""
import ctypes, sys

# 控制台若是 GBK, 下面的 ✅⚠ 会抛 UnicodeEncodeError → 统一 utf-8 输出
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def main():
    m = MS(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    GB = 1024.0 ** 3
    avail_commit = m.ullAvailPageFile / GB
    print('=' * 52)
    print(' 内存检查  (内存负载 %d%%)' % m.dwMemoryLoad)
    print('=' * 52)
    print(' 物理内存   可用 %6.1f GB / 共 %6.1f GB' % (m.ullAvailPhys / GB, m.ullTotalPhys / GB))
    print(' 提交内存   可用 %6.1f GB / 共 %6.1f GB   ← 关键' % (avail_commit, m.ullTotalPageFile / GB))
    print(' 进程VA     可用 %6.1f GB' % (m.ullAvailVirtual / GB))
    print('-' * 52)
    if avail_commit < 4:
        print(' ⛔ 提交内存几乎耗尽: 大型游戏/大模型会直接分配失败崩掉')
    elif avail_commit < 8:
        print(' ⚠ 提交内存偏紧: 建议先关掉不用的程序再进游戏')
    else:
        print(' ✅ 提交内存充足')
    print('-' * 52)
    print(' 谁在吃内存 (提交大小 Top10):')
    try:
        import subprocess
        ps = ('Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 10 '
              '@{n="p";e={$_.ProcessName}},@{n="MB";e={[int]($_.PrivateMemorySize64/1MB)}} | '
              'ForEach-Object { "   {0,-28} {1,8} MB" -f $_.p, $_.MB }')
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                             capture_output=True, text=True, timeout=30).stdout
        print(out.rstrip() or '   (读不到进程列表)')
    except Exception as e:
        print('   (读不到进程列表: %s)' % str(e)[:60])
    print('=' * 52)


if __name__ == '__main__':
    main()
