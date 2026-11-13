# -*- coding: utf-8 -*-
"""[2026-10-03] 助手开机自启：确保 server.py(8000) + 命名隧道 miao-assistant 在跑（幂等）"""
import os, socket, subprocess, sys, time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
PY = r'D:\ai\python\python.exe'
BASE = r'C:\\Users\\Jinqi Wei\\AppData\\Roaming\\AutoClaw-official\\accounts\\95f2263891c467552a767999f335d2750fae3cd30d13820fa1fcc2896984fe50\\projects\\Default\\smart-assistant'
CF = r'D:\ai\bin\cloudflared.exe'
WD = r'D:\ai\assistant_watchdog.py'
LOG = r'D:\ai\logs'
os.makedirs(LOG, exist_ok=True)

def busy(port):
    s = socket.socket(); s.settimeout(1)
    r = s.connect_ex(('127.0.0.1', port)); s.close()
    return r == 0

def running(match):
    # [2026-10-16] 加 watchdog 一类；判定口径与 assistant_watchdog.py 保持一致
    image, rx = {
        'server':   ('python.exe', 'server\\.py'),
        'miao':     ('cloudflared.exe', 'miao-assistant'),
        'watchdog': ('python.exe', 'assistant_watchdog\\.py'),
    }[match]
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='%s'\" | Where-Object { $_.CommandLine -match '%s' } | ForEach-Object { $_.ProcessId }"
           % (image, rx))
    # [2026-10-16] 无控制台父进程 spawn console 子进程会新建黑色窗口(桌面刷屏元凶)，一律禁窗
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0
    r = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True,
                       errors='replace', timeout=60, creationflags=0x08000000, startupinfo=si)
    return bool((r.stdout or "").strip())

def main():
    started = []
    if not busy(8000) or not running("server"):
        if not running("server"):
            subprocess.Popen([PY, 'server.py'], cwd=BASE,
                             stdout=open(os.path.join(LOG, 'assistant_server.log'), 'a', encoding='utf-8'),
                             stderr=subprocess.STDOUT, creationflags=0x00000008)
            started.append('server.py(8000)'); time.sleep(6)
    if not running('miao'):
        subprocess.Popen([CF, 'tunnel', 'run', 'miao-assistant'],
                         stdout=open(os.path.join(LOG, 'assistant_tunnel.log'), 'a', encoding='utf-8'),
                         stderr=subprocess.STDOUT, creationflags=0x00000008)
        started.append('tunnel'); time.sleep(8)
    # [2026-10-16] 看门狗以前从来没被拉起来过（日志只有 10-04 一行），公网失效只能靠人工
    if not running('watchdog'):
        subprocess.Popen([PY, WD], cwd=LOG,
                         stdout=open(os.path.join(LOG, 'assistant_watchdog.log'), 'a', encoding='utf-8'),
                         stderr=subprocess.STDOUT, creationflags=0x00000008)
        started.append('watchdog'); time.sleep(2)
    print('启动: ' + (', '.join(started) if started else '都已在跑') + ' | 8000=' + ('OK' if busy(8000) else 'FAIL')
          + ' | tunnel=' + ('OK' if running('miao') else 'FAIL') + ' | watchdog=' + ('OK' if running('watchdog') else 'FAIL'))
    return 0

sys.exit(main())
