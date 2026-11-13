# -*- coding: utf-8 -*-
"""通用引擎泄漏测试：找引擎进程（非面板，可加命令行过滤）→ 等加载 → 硬杀 → 量归还
用法: python kill_test_any.py [阈值GB] [等待分钟] [命令行关键字]"""
import ctypes, os, subprocess, sys, time
OUT = os.path.join(os.environ.get('TEMP', '.'), 'kill_test_any2.txt'); o = []
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
WAIT_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
KEY = sys.argv[3] if len(sys.argv) > 3 else ''
class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
def cm():
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPageFile - m.ullAvailPageFile)/2**30, m.ullAvailPhys/2**30
PS = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -notmatch 'engine_manager|kill_test' } | Sort-Object PrivatePageCount -Descending | ForEach-Object { $_.ProcessId.ToString() + '|' + [math]::Round($_.PrivatePageCount/1GB,2) + '|' + $_.CommandLine }")
def find_engine():
    r = subprocess.run(['powershell', '-NoProfile', '-Command', PS], capture_output=True, text=True, timeout=60)
    for line in (r.stdout or '').split('\n'):
        parts = line.strip().split('|', 2)
        if len(parts) == 3:
            if KEY and KEY.lower() not in parts[2].lower():
                continue
            return parts
    return None
b, bf = cm()
o.append('起始 %.1f GB | 物理可用 %.1f | 阈值 %.0f GB | 等 %d 分钟 | 过滤=%s' % (b, bf, TH, WAIT_MIN, KEY or '无'))
target = None
for i in range(WAIT_MIN * 6):
    p = find_engine()
    if p:
        pid, commit, cmd = p
        c, f = cm()
        o.append('  [%s] PID %s 提交 %s GB | 系统 %.1f GB | %s' % (time.strftime('%H:%M:%S'), pid, commit, c, cmd[:60]))
        if float(commit) >= TH:
            target = (pid, commit, cmd); break
    else:
        o.append('  [%s] 没有匹配的引擎进程' % time.strftime('%H:%M:%S'))
    time.sleep(10)
if target:
    pid, commit, cmd = target
    c_before, _ = cm()
    o.append('--- 加载完成 PID %s 提交 %s GB → 硬杀 ---' % (pid, commit))
    o.append('  进程: %s' % cmd[:90])
    r = subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, text=True, errors='replace', timeout=30)
    o.append('  taskkill: %s' % ((r.stdout or r.stderr or '').strip()[:80]))
    time.sleep(3); c1, _ = cm()
    o.append('  杀后 3 秒: %.1f GB (%+.1f)' % (c1, c1 - c_before))
    time.sleep(20); c2, f2 = cm()
    o.append('  杀后 23 秒: %.1f GB (%+.1f) | 物理可用 %.1f GB' % (c2, c2 - c_before, f2))
    o.append('  判定: %s' % ('归还 OK' if (c2 - c_before) < 3 else '残留 %.1f GB' % (c2 - c_before)))
else:
    o.append('未等到加载完成（未杀任何进程）')
open(OUT, 'w', encoding='utf-8').write('\n'.join(o)); print('done')
