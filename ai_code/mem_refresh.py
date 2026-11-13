# -*- coding: utf-8 -*-
"""[2026-10-03] 一键清理虚拟内存挂账：模拟 Win+Ctrl+Shift+B（重启显卡驱动，不丢数据）+ 前后对比提交量"""
import ctypes, os, subprocess, sys, time
OUT = r'D:\ai\logs\mem_refresh.log'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
def snap():
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return (m.ullTotalPageFile - m.ullAvailPageFile)/2**30, m.ullAvailPhys/2**30
def busy():
    """有引擎/python 任务在跑就别刷新（会打断任务）"""
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { $_.CommandLine -match 'flux|gen_image|ltx|llama|qwen|comfy' } | Measure-Object | Select-Object -ExpandProperty Count"],
                           capture_output=True, text=True, timeout=30)
        return int((r.stdout or '0').strip() or 0) > 0
    except Exception:
        return False
c0, f0 = snap()
print('刷新前: 已提交 %.1f GB | 物理可用 %.1f GB' % (c0, f0))
if busy():
    print('⚠ 检测到引擎/生成任务在跑 ⇒ 不做刷新（避免打断任务）。等空闲再运行。')
    open(OUT, 'a', encoding='utf-8').write('%s  SKIP(忙) 提交%.1f free%.1f\n' % (time.strftime('%F %T'), c0, f0))
    sys.exit(2)
VK = {'LWIN': 0x5B, 'CTRL': 0x11, 'SHIFT': 0x10, 'B': 0x42}
for k in ('LWIN', 'CTRL', 'SHIFT'):
    ctypes.windll.user32.keybd_event(VK[k], 0, 0, 0)
ctypes.windll.user32.keybd_event(VK['B'], 0, 0, 0)
time.sleep(0.05)
for k in ('B', 'SHIFT', 'CTRL', 'LWIN'):
    ctypes.windll.user32.keybd_event(VK[k], 0, 2, 0)
time.sleep(6)
c1, f1 = snap()
print('刷新后: 已提交 %.1f GB（%+.1f）| 物理可用 %.1f GB（%+.1f）' % (c1, c1-c0, f1, f1-f0))
open(OUT, 'a', encoding='utf-8').write('%s  提交%.1f→%.1f (%+.1f)  free%.1f→%.1f (%+.1f)\n' % (time.strftime('%F %T'), c0, c1, c1-c0, f0, f1, f1-f0))
