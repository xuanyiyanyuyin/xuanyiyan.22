# -*- coding: utf-8 -*-
"""[2026-10-03] 保证助手公网隧道在跑，并把"当前网址"实时写盘（桌面 + D:\ai + 项目 data）"""
import os, re, subprocess, sys, time, json

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CF = r'D:\ai\bin\cloudflared.exe'
LOCAL = 'http://127.0.0.1:8000'
LOG = os.path.join(os.environ.get('TEMP', '.'), 'sa_tunnel.log')
ERR = LOG + '.err'
DESK = os.path.join(os.environ['USERPROFILE'], 'Desktop')
OUTS = [r'D:\ai\当前网址.txt',
        os.path.join(DESK, '助手_当前网址.txt'),
        r'C:\Users\Jinqi Wei\AppData\Roaming\AutoClaw-official\accounts\95f2263891c467552a767999f335d2750fae3cd30d13820fa1fcc2896984fe50\projects\Default\smart-assistant\data\当前网址.txt']


def port_busy(p=8000):
    import socket
    s = socket.socket(); s.settimeout(1)
    r = s.connect_ex(('127.0.0.1', p)); s.close(); return r == 0


def tunnel_pids():
    r = subprocess.run(['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\" | Where-Object { $_.CommandLine -match '--url http://127.0.0.1:8000' } | ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True, errors='replace', timeout=60)
    return [l.strip() for l in (r.stdout or '').split(chr(10)) if l.strip()]


def read_url():
    import glob as _g
    cands = [LOG, ERR] + _g.glob(os.path.join(os.environ.get('TEMP', '.'), 'sa_tunnel*'))
    for f in cands:
        try:
            t = open(f, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        m = re.findall('https://[a-z0-9-]+' + chr(92) + '.trycloudflare' + chr(92) + '.com', t)
        if m:
            return m[-1]
    return ''


def ensure_tunnel():
    if not tunnel_pids():
        open(LOG, 'w').close()
        subprocess.Popen([CF, 'tunnel', '--no-autoupdate', '--url', LOCAL],
                         stdout=open(LOG, 'a', encoding='utf-8'),
                         stderr=open(ERR, 'a', encoding='utf-8'),
                         creationflags=0x00000008)
        for _ in range(30):
            time.sleep(2)
            if read_url():
                break
    return read_url()


def main():
    if not port_busy():
        print('❌ 本机 8000 没在跑（助手服务没启动）')
        return 2
    url = ensure_tunnel()
    if not url:
        print('⚠ 还没拿到公网地址（隧道可能还在建立），稍后重跑')
        return 3
    txt = ('暄依嫣 · 智能小助手  当前公网地址\n'
           + '=' * 46 + '\n' + url + '\n' + '=' * 46 + '\n'
           + '更新时间: ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n'
           + '说明: 这是临时隧道地址，重启隧道/重启电脑后会变；\n'
           + '      跑一次 D:\\ai\\助手公网_刷新网址.bat 就能拿到最新地址。\n'
           + '      本机后台(仅你能进): ' + LOCAL + '\n')
    for p in OUTS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w', encoding='utf-8').write(txt)
        except Exception as e:
            print('  写入失败 %s: %s' % (p, str(e)[:60]))
    print(url)
    return 0


sys.exit(main())
