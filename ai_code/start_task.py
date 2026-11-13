# -*- coding: utf-8 -*-
"""用法: python start_task.py <引擎名> [key=value ...]（避免 JSON 转义问题）"""
import json, sys, urllib.request
name = sys.argv[1]
params = {}
for kv in sys.argv[2:]:
    if '=' in kv:
        k, v = kv.split('=', 1)
        try:
            params[k] = int(v)
        except ValueError:
            params[k] = v
req = urllib.request.Request('http://127.0.0.1:8321/api/task',
                             data=json.dumps({'name': name, 'params': params}).encode(),
                             headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(req, timeout=30).read().decode()[:160])
