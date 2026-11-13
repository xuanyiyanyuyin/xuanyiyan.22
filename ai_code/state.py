# -*- coding: utf-8 -*-
import ctypes, subprocess, socket
class MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
m = MS(); m.dwLength = ctypes.sizeof(MS)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print('已提交 %.1f GB / 161.8 | 物理可用 %.1f GB' % ((m.ullTotalPageFile - m.ullAvailPageFile)/2**30, m.ullAvailPhys/2**30))
r = subprocess.run(['powershell','-NoProfile','-Command',
  "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { $_.ProcessId.ToString() + '  ' + [math]::Round($_.PrivatePageCount/1GB,2) + ' GB  ' + $_.CommandLine.Substring(0,[Math]::Min(70,$_.CommandLine.Length)) }"],
  capture_output=True, text=True, errors='replace', timeout=60)
for line in (r.stdout or '').split('\n'):
    if line.strip():
        print('  PID ' + line.strip())
for p in (8080, 8081, 8188, 8321, 8330):
    sk = socket.socket(); sk.settimeout(0.5); ok = sk.connect_ex(('127.0.0.1', p)) == 0; sk.close()
    print('  端口 %d: %s' % (p, '在听' if ok else '未监听'))
