#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[2026-10-02] 喵朵双通道切换：NapCat(自建QQ客户端) ↔ 官方QQ机器人

两者都占 3000 端口 ⇒ 不能同时跑。本脚本负责停一个、起另一个。
用法（用聊天栈 python）：
  python qq_channel.py status            # 看当前是谁在跑
  python qq_channel.py official          # 切到官方通道（停 NapCat，起网关）
  python qq_channel.py napcat            # 切回 NapCat 通道（停网关，起 NapCat）
"""
import os
import subprocess
import sys
import time

PYL = r'D:\ai\python_ltx25\Scripts\python.exe'
PYM = r'D:\ai\python\python.exe'
GATEWAY = r'D:\ai\qq_official_gateway.py'
QQ_CONTROL = r'D:\ai\llamacpp\qq_control.py'
LOG_DIR = r'D:\ai\llamacpp'
NAP_MARK = ('napcatwinbootmain.exe', 'QQ.exe')


def port_owner(port=3000):
    """按端口找属主 PID（本机 Get-Process 枚举不可靠，端口法最稳）"""
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "(Get-NetTCPConnection -LocalPort %d -State Listen -ErrorAction SilentlyContinue).OwningProcess" % port],
                             capture_output=True, text=True, timeout=30).stdout
        pids = [int(x) for x in out.split() if x.strip().isdigit()]
        return pids
    except Exception:
        return []


def kill_pid(pid):
    try:
        subprocess.run(['powershell', '-NoProfile', '-Command', 'Stop-Process -Id %d -Force' % pid],
                       capture_output=True, timeout=30)
        return True
    except Exception:
        return False


def procs_by_mark(marks):
    """按命令行特征找进程（精确 PID，不用 pkill -f 那种模糊匹配）"""
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | "
                              "ForEach-Object { \"$($_.ProcessId)|$($_.Name)|$($_.CommandLine)\" }"],
                             capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split('|', 2)
        if len(parts) < 3:
            continue
        pid, name, cmd = parts
        if any(m.lower() in cmd.lower() or m.lower() == name.lower() for m in marks):
            try:
                rows.append((int(pid), name, cmd[:80]))
            except Exception:
                pass
    return rows


def gw_alive():
    return bool(port_owner(3000)) and not napcat_alive()


def napcat_alive():
    return bool(procs_by_mark(NAP_MARK))


def status():
    owner = port_owner(3000)
    nap = procs_by_mark(NAP_MARK)
    gw = [r for r in procs_by_mark(('qq_official_gateway.py',))]
    print('3000 端口属主 PID: %s' % (owner or '无'))
    print('官方网关进程: %s' % (gw or '未运行'))
    print('NapCat/QQ 进程: %s' % (nap or '未运行'))
    if gw and not nap:
        print('=> 当前通道: 官方 QQ 机器人')
    elif nap and not gw:
        print('=> 当前通道: NapCat')
    elif not gw and not nap:
        print('=> 两条通道都没跑（8899/8081 可能还在，但收不到 QQ 消息）')
    else:
        print('=> ⚠ 两条通道都在跑（会抢 3000，需要处理）')


def start_official():
    print('启动官方网关…')
    subprocess.Popen([PYL, '-u', GATEWAY], cwd=r'D:\ai',
                     stdout=open(os.path.join(os.environ['TEMP'], 'gw.out'), 'ab'),
                     stderr=open(os.path.join(os.environ['TEMP'], 'gw.err'), 'ab'))
    for _ in range(15):
        time.sleep(1)
        if port_owner(3000):
            print('官方网关已监听 3000 ✅')
            return True
    print('网关未在 15 秒内监听 3000，请看 %TEMP%\\gw.out')
    return False


def start_napcat():
    print('启动 NapCat（走 qq_control）…')
    subprocess.run([PYL, QQ_CONTROL, 'start'], cwd=r'D:\ai\llamacpp', timeout=600)
    return True


def stop_official():
    for pid in port_owner(3000):
        print('停官方网关 PID=%d' % pid)
        kill_pid(pid)


def stop_napcat():
    print('停 NapCat（qq_control stop 会一并停其它服务，这里只优雅退 NapCat）…')
    try:
        import urllib.request
        urllib.request.urlopen('http://127.0.0.1:3000/bot_exit', timeout=5).read()
        time.sleep(6)
    except Exception:
        pass
    for pid, name, cmd in procs_by_mark(NAP_MARK):
        print('结束 %s PID=%d' % (name, pid))
        kill_pid(pid)


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    act = (sys.argv[1] if len(sys.argv) > 1 else 'status').lower()
    if act == 'status':
        status()
    elif act == 'official':
        stop_napcat()
        time.sleep(2)
        start_official()
        status()
    elif act == 'napcat':
        stop_official()
        time.sleep(2)
        start_napcat()
        status()
    else:
        print('用法: python qq_channel.py status|official|napcat')
