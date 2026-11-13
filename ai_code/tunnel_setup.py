# -*- coding: utf-8 -*-
"""[2026-10-03] 建 Named Tunnel 并绑定 assistant.xuanyiyan.cn（永久地址）"""
import json, os, subprocess, sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CF = r'D:\ai\bin\cloudflared.exe'
NAME = 'miao-assistant'
HOST = 'assistant.xuanyiyan.cn'
LOCAL = 'http://127.0.0.1:8000'
HOME = os.path.expanduser('~')
CFDIR = os.path.join(HOME, '.cloudflared')
CFG = os.path.join(CFDIR, 'config.yml')

def run(args, **kw):
    print('> cloudflared ' + ' '.join(args))
    return subprocess.run([CF] + args, capture_output=True, text=True, errors='replace', **kw)

os.makedirs(CFDIR, exist_ok=True)
if not os.path.exists(os.path.join(CFDIR, 'cert.pem')):
    print('缺少 cert.pem ⇒ 先跑:  cloudflared tunnel login（浏览器里授权 xuanyiyan.cn）')
    sys.exit(2)

r = run(['tunnel', 'create', NAME])
out = (r.stdout or '') + (r.stderr or '')
print(out.strip()[:400])
uid = None
for tok in out.replace('=', ' ').split():
    if len(tok) == 36 and tok.count('-') == 4:
        uid = tok
if not uid:
    lst = run(['tunnel', 'list', '--output', 'json'])
    try:
        for t in json.loads(lst.stdout or '[]'):
            if t.get('name') == NAME:
                uid = t.get('id')
    except Exception:
        pass
if not uid:
    print('未能拿到 tunnel UUID；把上面输出发给暄依嫣')
    sys.exit(3)
print('tunnel id = ' + uid)

run(['tunnel', 'route', 'dns', NAME, HOST])

open(CFG, 'w', encoding='utf-8').write(
    'tunnel: %s\n' % NAME +
    'credentials-file: %s\n' % os.path.join(CFDIR, uid + '.json') +
    'ingress:\n' +
    '  - hostname: %s\n' % HOST +
    '    service: %s\n' % LOCAL +
    '  - service: http_status:404\n')
print('已写 config: ' + CFG)
print('下一步: 双击 D:\\ai\\assistant_tunnel.bat')
