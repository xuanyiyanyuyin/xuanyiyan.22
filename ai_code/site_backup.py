# -*- coding: utf-8 -*-
"""[2026-10-04] 站点一键备份到 GitHub 私人库（走 api.github.com，绕开被墙的 git 传输）
用法: D:\ai\python\python.exe D:\ai\site_backup.py [提交说明]
说明: token 从 D:\ai\.gh_token 读取（也可用环境变量 GH_TOKEN 覆盖）"""
import base64, datetime, json, os, sys, time, urllib.request
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
REPO = "/repos/xuanyiyanyuyin/xuan-assistant"
B = r"C:\Users\Jinqi Wei\AppData\Roaming\AutoClaw-official\accounts\95f2263891c467552a767999f335d2750fae3cd30d13820fa1fcc2896984fe50\projects\Default\smart-assistant"
TOK = os.environ.get("GH_TOKEN") or open(r"D:\ai\.gh_token", encoding="utf-8").read().strip()
P = os.environ.get("HTTPS_PROXY") or "127.0.0.1:7892"
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({"http": P, "https": P})))
def api(path, method="GET", body=None, tries=4):
    last = None
    for k in range(tries):
        try:
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request("https://api.github.com" + REPO + path, data=data, method=method, headers={
                "Authorization": "token " + TOK, "Accept": "application/vnd.github+json",
                "User-Agent": "site-backup", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e; time.sleep(2 + 2 * k)
    raise last
msg = sys.argv[1] if len(sys.argv) > 1 else ("站点自动备份 " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
head = api("/git/ref/heads/main")["object"]["sha"]
files, n = [], 0
for root, dirs, fs in os.walk(B):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    for f in fs:
        if f.endswith(".pyc") or ".bak_before_" in f: continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, B).replace(chr(92), "/")
        blob = api("/git/blobs", "POST", {"content": base64.b64encode(open(p, "rb").read()).decode(), "encoding": "base64"})
        files.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        n += 1
tree = api("/git/trees", "POST", {"tree": files})
commit = api("/git/commits", "POST", {"message": msg, "tree": tree["sha"], "parents": [head]})
api("/git/refs/heads/main", "PATCH", {"sha": commit["sha"], "force": True})
print("备份完成: %d 个文件 -> 提交 %s（%s）" % (n, commit["sha"][:8], msg))