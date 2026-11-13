#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[2026-10-02] 喵朵总开关：暂停 / 恢复 / 全量群消息模式切换

开关用「文件存在与否」表示 ⇒ 改完立刻生效，不用重启机器人。
  python miao_ctrl.py status    查看状态
  python miao_ctrl.py pause     喵朵完全静默（群聊+私聊都不理）
  python miao_ctrl.py resume    恢复
  python miao_ctrl.py nofull    只在被 @ 或被提起时回（关掉全量模式）
  python miao_ctrl.py full      开回全量模式（喵朵自己判断回不回）
  python miao_ctrl.py off       彻底下线：暂停 + 停官方网关（连消息都收不到）
  python miao_ctrl.py on        off 的逆操作
"""
import os
import subprocess
import sys
import time

FLAG_DIR = r'D:\ai\llamacpp\memory'
PAUSE = os.path.join(FLAG_DIR, 'miaoduo_pause.flag')
FULL_OFF = os.path.join(FLAG_DIR, 'group_full_off.flag')
PYL = r'D:\ai\python_ltx25\Scripts\python.exe'
GATEWAY = r'D:\ai\qq_official_gateway.py'


def port_owner(port=3000):
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "(Get-NetTCPConnection -LocalPort %d -State Listen -ErrorAction SilentlyContinue).OwningProcess" % port],
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def touch(p, on=True):
    try:
        if on:
            open(p, 'a', encoding='utf-8').close()
        elif os.path.exists(p):
            os.remove(p)
    except Exception as e:
        print('操作失败 %s: %s' % (p, e))
        return False
    return True


def status():
    print('暂停开关   miaoduo_pause.flag     : %s' % ('存在（喵朵静默）' if os.path.exists(PAUSE) else '不存在（正常说话）'))
    print('全量群消息 group_full_off.flag   : %s' % ('存在（只在被@/提起时回）' if os.path.exists(FULL_OFF) else '不存在（全量模式，喵朵自己判断）'))
    print('官方网关 3000                    : %s' % ('在跑' if port_owner(3000) else '没跑（收不到 QQ 消息）'))
    for pt, name in ((8081, '代理'), (8899, '喵朵适配层')):
        c = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "(Test-NetConnection -ComputerName 127.0.0.1 -Port %d -InformationLevel Quiet -WarningAction SilentlyContinue)" % pt],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        print('%-6s %-26d: %s' % (name, pt, '在跑' if c.lower().startswith('true') else '没跑'))


def gw_stop():
    for pid in port_owner(3000):
        subprocess.run(['powershell', '-NoProfile', '-Command', 'Stop-Process -Id %d -Force' % pid],
                       capture_output=True, timeout=30)
        print('已停官方网关 PID=%d' % pid)


def gw_start():
    if port_owner(3000):
        print('官方网关已在跑')
        return
    subprocess.Popen([PYL, '-u', GATEWAY], cwd=r'D:\ai',
                     stdout=open(os.path.join(os.environ['TEMP'], 'gw.out'), 'ab'),
                     stderr=open(os.path.join(os.environ['TEMP'], 'gw.err'), 'ab'))
    for _ in range(15):
        time.sleep(1)
        if port_owner(3000):
            print('官方网关已启动 ✅')
            return
    print('网关 15 秒内没起来，看 %TEMP%\\gw.out')


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    act = (sys.argv[1] if len(sys.argv) > 1 else 'status').lower()
    if act == 'pause':
        touch(PAUSE, True); print('喵朵已暂停（静默）')
    elif act == 'resume':
        touch(PAUSE, False); print('喵朵已恢复')
    elif act == 'nofull':
        touch(FULL_OFF, True); print('已关全量模式：只在被 @ / 被提起时回')
    elif act == 'full':
        touch(FULL_OFF, False); print('已开全量模式：喵朵自己判断回不回')
    elif act == 'off':
        touch(PAUSE, True); gw_stop(); print('喵朵已彻底下线（暂停 + 网关已停）')
    elif act == 'on':
        touch(PAUSE, False); gw_start(); print('喵朵已上线')
    status()
