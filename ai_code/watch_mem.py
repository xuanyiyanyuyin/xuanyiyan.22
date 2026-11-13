r"""内存监视: 每 30 秒往日志里记一次「提交内存」和吃内存最多的进程
用途: 挂着它去玩游戏/跑引擎, 事后看是谁把提交内存吃光的。

用法:
    python D:\ai\watch_mem.py                # 跟命令行同生共死 (Ctrl+C 停)
    python D:\ai\watch_mem.py 15             # 每 15 秒采一次
日志: D:\ai\watch_mem.log  (UTF-8, 超过 8MB 自动轮转)
"""
import ctypes, os, sys, time, datetime

LOG = r'D:\ai\watch_mem.log'
MAXLOG = 8 * 1024 * 1024

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def snapshot():
    m = MS(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    GB = 1024.0 ** 3
    return (m.ullTotalPageFile - m.ullAvailPageFile) / GB, m.ullAvailPageFile / GB, m.ullAvailPhys / GB


def top_procs(n=6):
    """用 WMI 取提交最大的几个进程 (不依赖 psutil)"""
    rows = []
    try:
        import subprocess
        cmd = ('Get-CimInstance Win32_Process | Sort-Object PageFileUsage -Descending | '
               'Select-Object -First %d | ForEach-Object { "{0}|{1}|{2}" -f $_.Name,$_.ProcessId,'
               '[int]($_.PageFileUsage/1KB) }' % n)
        out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd],
                             capture_output=True, text=True, timeout=40).stdout
        for line in out.splitlines():
            p = line.strip().split('|')
            if len(p) == 3:
                rows.append('%s(%s)=%sMB' % (p[0], p[1], p[2]))
    except Exception as e:
        rows.append('(读进程失败 %s)' % str(e)[:40])
    return rows


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    if os.path.exists(LOG) and os.path.getsize(LOG) > MAXLOG:
        os.replace(LOG, LOG + '.old')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('\n===== 开始监视: %s (每 %d 秒) =====\n' % (datetime.datetime.now(), interval))
        f.flush()
        print('监视中… 日志: %s   (Ctrl+C 停止)' % LOG)
        while True:
            used, free_commit, free_phys = snapshot()
            line = ('[%s] 提交已用=%6.1fGB 可用提交=%6.1fGB 可用物理=%5.1fGB | %s'
                    % (datetime.datetime.now().strftime('%m-%d %H:%M:%S'), used, free_commit,
                       free_phys, ' ; '.join(top_procs())))
            f.write(line + '\n'); f.flush()
            print(line)
            if free_commit < 3:
                f.write('  !! 可用提交已不足 3GB: 再进游戏/加载模型铁定失败\n'); f.flush()
            time.sleep(interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已停止')
