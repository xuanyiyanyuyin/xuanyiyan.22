# -*- coding: utf-8 -*-
"""[2026-10-16 修复] 助手站点看门狗：每 30 秒一轮，管两件旧版没管的事。
旧版三处失效：
  ① 只用 TCP connect_ex 探 8000 —— 进程僵住但端口还在监听时，它认为一切正常，救不回来；
  ② 只管 server.py，不管 cloudflared —— 隧道会自己 "no more connections active and exiting"，
     那时 8000 健康、assistant.xuanyiyan.cn 却是 502/1033；
  ③ 它自己没进开机自启（日志只有 10-04 一行），所以①②从来没生效过。
本版：真打 /api/health 判活 + 守命名隧道 miao-assistant + 只杀本看门狗自己拉起过的进程。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = r"C:\Users\Jinqi Wei\AppData\Roaming\AutoClaw-official\accounts\95f2263891c467552a767999f335d2750fae3cd30d13820fa1fcc2896984fe50\projects\Default\smart-assistant"
PY = r"D:\ai\python\python.exe"
CF = r"D:\ai\bin\cloudflared.exe"
LOGDIR = r"D:\ai\logs"
SRVLOG = os.path.join(LOGDIR, "assistant_server.log")
TUNLOG = os.path.join(LOGDIR, "assistant_tunnel.log")
WDLOG = os.path.join(LOGDIR, "assistant_watchdog.log")
OWNED = os.path.join(os.environ.get("TEMP", "."), "assistant_watchdog_owned.json")
HEALTH = "http://127.0.0.1:8000/api/health"
TUNNEL = "miao-assistant"
INTERVAL = 30
DEAD_LIMIT = 3          # 连续 3 轮(~90s)不应答才动手，避免误伤正在启动的进程
# [2026-10-16 修复] 本进程是 DETACHED 无控制台启动的，任何 console 子进程(powershell/tasklist)
# 都会被 Windows 新建一个黑色控制台窗口 —— 每 30 秒弹两次就是刚才桌面刷屏的元凶。
# 所有子进程一律 CREATE_NO_WINDOW + STARTF_USESHOWWINDOW 双保险。
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW

def _si():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si

_ps = ["powershell", "-NoProfile", "-Command"]


def log(msg):
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(WDLOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass


def health_ok(timeout=6):
    """真打一次健康接口：进程僵住/还在初始化都不算活。"""
    try:
        with urllib.request.urlopen(HEALTH, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return str(d.get("status")) == "ok"
    except Exception:
        return False


def _pids(image, cmd_regex):
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='%s'\" | Where-Object { $_.CommandLine -match '%s' } "
           "| ForEach-Object { $_.ProcessId }") % (image, cmd_regex)
    try:
        r = subprocess.run(_ps + [cmd], capture_output=True, text=True, errors="replace", timeout=60,
                           creationflags=NO_WINDOW, startupinfo=_si())
        return [int(x) for x in (r.stdout or "").split() if x.strip().isdigit()]
    except Exception:
        return []


def server_pids():
    return _pids("python.exe", "server\\.py")


def tunnel_pids():
    return _pids("cloudflared.exe", TUNNEL)


def _load_owned():
    try:
        return set(int(x) for x in json.load(open(OWNED, encoding="utf-8")))
    except Exception:
        return set()


def _save_owned(s):
    try:
        json.dump(sorted(s), open(OWNED, "w", encoding="utf-8"))
    except Exception:
        pass


def _spawn(args, cwd, logfile, tag):
    try:
        p = subprocess.Popen(args, cwd=cwd, stdout=open(logfile, "a", encoding="utf-8"),
                             stderr=subprocess.STDOUT, creationflags=NO_WINDOW, startupinfo=_si())
    except Exception as e:
        log("拉起 %s 异常: %s" % (tag, str(e)[:120]))
        return 0
    o = _load_owned()
    o.add(p.pid)
    _save_owned(o)
    log("拉起 %s -> pid %d" % (tag, p.pid))
    return p.pid


def kill(pid):
    try:
        subprocess.run(_ps + ["Stop-Process -Id %d -Force" % pid], capture_output=True, timeout=60,
                       creationflags=NO_WINDOW, startupinfo=_si())
        log("已结束僵死进程 pid %d" % pid)
        return True
    except Exception as e:
        log("结束 pid %d 失败: %s" % (pid, str(e)[:100]))
        return False


def one_round(dead, miss_t):
    # ---- 隧道：探测一次为空不算掉线（powershell 在高负载下会超时返回空，
    #      15:32 就是这样误拉起了一条重复隧道），必须连续 2 轮都为空才补 ----
    if tunnel_pids():
        miss_t = 0
    else:
        miss_t += 1
        if miss_t >= 2:
            log("连续 %d 轮未见 %s 隧道 -> 拉起" % (miss_t, TUNNEL))
            _spawn([CF, "tunnel", "run", TUNNEL], LOGDIR, TUNLOG, "tunnel")
            miss_t = 0

    # ---- 服务 ----
    if health_ok():
        return 0, miss_t
    sp = server_pids()
    if not sp:
        log("健康检查失败且无 server.py 进程 -> 拉起")
        _spawn([PY, "server.py"], BASE, SRVLOG, "server.py(8000)")
        return 0, miss_t
    if dead < DEAD_LIMIT:
        return dead + 1, miss_t
    owned = _load_owned()
    wedged = [p for p in sp if p in owned]
    if not wedged:
        # 不代杀别人拉起的进程：可能是主人手动跑的，顶多重启不了也不能抢
        log("⚠ server.py(pid %s) 在监听但不应答，且不是本看门狗拉起的 —— 不代杀，需主人处理" % ",".join(map(str, sp)))
        return 0, miss_t
    log("连续 %d 轮不应答 -> 结束 pid %s 后重拉" % (dead, wedged[0]))
    kill(wedged[0])
    time.sleep(2)
    _spawn([PY, "server.py"], BASE, SRVLOG, "server.py(8000)")
    return 0, miss_t


def main():
    log("watchdog 启动 (健康检查=%s, 隧道=%s, 周期=%ds, 禁窗=ON)" % (HEALTH, TUNNEL, INTERVAL))
    # 开机自启链先起的那个 server.py 不由本进程拉起，这里接管它，之后才救得回来；
    # 看门狗起来之后主人才手动另起的进程不在名单里，一律不代杀。
    boot = server_pids()
    if boot:
        o = _load_owned() | set(boot)
        _save_owned(o)
        log("接管已在跑的 server.py pid %s" % ",".join(map(str, boot)))
    dead = 0
    miss_t = 0
    while True:
        time.sleep(INTERVAL)
        try:
            dead, miss_t = one_round(dead, miss_t)
        except Exception as e:
            log("本轮异常: " + str(e)[:150])
    return 0


if __name__ == "__main__":
    sys.exit(main())
