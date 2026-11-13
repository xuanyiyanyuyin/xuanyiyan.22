# -*- coding: utf-8 -*-
"""一键关闭：AI 引擎 / QQ 机器人（按端口 + 进程特征双保险，关完自检）

用法:
    python D:\\ai\\stop_all.py ai          关 AI 引擎与面板 (8321 / 8081 / 8123 / 8188 + 引擎进程)
    python D:\\ai\\stop_all.py qq          关 QQ 机器人   (NapCat/QQ注入进程 + 3000 / 6099 / 8899 / 8081 / 8321)
    python D:\\ai\\stop_all.py all         两个都关（默认）
    python D:\\ai\\stop_all.py all --dry   只报告"将会关掉谁"，不动手

注意: 关 QQ 机器人 = NapCat 退出登录，下次启动可能需要重新扫码。
"""
import os
import re
import socket
import subprocess
import sys
import time

try:
    # 2026-09-18 修复: 不再强制 UTF-8 —— 桌面 bat 现在用 chcp 936(GBK 控制台),
    # 强制 UTF-8 会让中文变乱码。这里只放宽错误处理, 编码跟随控制台。
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

CREATE_NO_WINDOW = 0x08000000

PORTS = {
    'ai': [
        (8321, 'engine_manager 引擎面板'),
        (8081, 'websearch_proxy 代理'),
        (8123, 'v4flash 推理服务'),
        (8188, 'ComfyUI'),
    ],
    'qq': [
        (3000, 'NapCat OneBot'),
        (6099, 'NapCat WebUI'),
        (8899, 'qq_bot 适配层'),
        (8081, 'websearch_proxy 代理'),
        (8321, 'engine_manager 面板'),
    ],
}

# 进程命令行特征（只看"脚本/参数"部分，不看解释器路径，避免 python_ltx25 误伤）
PY_PATTERNS_AI = [
    'engine_manager', 'websearch_proxy', 'v4flash', 'qwen36_server', 'qwen_server',
    'llama-server', 'llama_server', 'gen_image', 'gen_video', 'gen_audio',
    'flux_3tier', 'flux2_3tier', 'ltx_upscale', 'ltx_video', 'comfy',
    'range_server', 'upscale4x',
]
PY_PATTERNS_QQ = ['qq_bot']
# 非 python 的引擎/框架进程（按进程名）
EXE_NAMES_AI = ['llama-server.exe', 'llama-server', 'koboldcpp.exe']


def port_pids(port):
    """返回监听该端口的 pid 列表"""
    try:
        out = subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True, text=True,
                             timeout=20, creationflags=CREATE_NO_WINDOW).stdout
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        if 'LISTENING' in line and re.search(r'[:.]%d\s' % port, line):
            parts = line.split()
            if parts and parts[-1].isdigit() and parts[-1] not in pids:
                pids.append(parts[-1])
    return pids


def proc_list():
    """返回 [(pid, name, cmdline)]"""
    try:
        import json as _json
        ps = ('Get-CimInstance Win32_Process | '
              'Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress')
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True,
                             text=True, timeout=40, creationflags=CREATE_NO_WINDOW).stdout.strip()
        data = _json.loads(out) if out else []
        if isinstance(data, dict):
            data = [data]
        return [(int(d.get('ProcessId') or 0), d.get('Name') or '', d.get('CommandLine') or '') for d in data]
    except Exception as e:
        print('  (进程列表读取失败: %s)' % str(e)[:60])
        return []


def targets(mode):
    """收集要关的目标: [(kind, key, pid, 说明)]"""
    out, seen = [], set()
    for m in (['ai', 'qq'] if mode == 'all' else [mode]):
        for port, label in PORTS.get(m, []):
            for pid in port_pids(port):
                k = ('pid', pid)
                if k in seen:
                    continue
                seen.add(k)
                out.append(('端口 %-5d' % port, label, pid))
    # 进程特征
    for pid, name, cmd in proc_list():
        n = (name or '').lower()
        c = cmd or ''
        # 只取解释器之后的参数部分（避免 python_ltx25 这种路径命中特征词）
        parts = c.split()
        args = ' '.join(parts[1:]) if len(parts) > 1 else ''
        al = args.lower()
        pats = list(PY_PATTERNS_AI) + (list(PY_PATTERNS_QQ) if mode in ('qq', 'all') else [])
        hit = None
        if n == 'python.exe' and any(p in al for p in pats):
            hit = 'python 进程 (%s)' % next(p for p in pats if p in al)
        elif n in [x.lower() for x in EXE_NAMES_AI]:
            hit = '推理服务 %s' % name
        elif mode in ('qq', 'all') and n == 'napcatwinbootmain.exe':
            hit = 'NapCat 启动器'
        elif mode in ('qq', 'all') and n == 'qq.exe' and 'enable-logging' in al:
            hit = 'QQ（NapCat 注入实例，不是你的日常 QQ）'
        if hit:
            k = ('pid', pid)
            if k in seen:
                continue
            seen.add(k)
            out.append(('进程 %-7s' % pid, hit, pid))
    return out


def kill(pid):
    try:
        r = subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True, text=True,
                           timeout=20, creationflags=CREATE_NO_WINDOW)
        return r.returncode == 0
    except Exception:
        return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    dry = any(a in ('--dry', '-n', 'dry') for a in sys.argv[1:])
    mode = (args[0] if args else 'all').lower()
    if mode not in ('ai', 'qq', 'all'):
        print(__doc__)
        return 1

    title = {'ai': 'AI 引擎与面板', 'qq': 'QQ 机器人', 'all': 'AI 引擎 + QQ 机器人'}[mode]
    print('=== 一键关闭：%s%s ===' % (title, '（预演模式，不会真关）' if dry else ''))
    tg = targets(mode)
    if not tg:
        print('  没有发现需要关闭的进程/端口 —— 已经是关着的状态 ✓')
        return 0

    print('  将要关闭 %d 个目标:' % len(tg))
    for kind, label, pid in tg:
        print('    %s  pid=%-7s %s' % (kind, pid, label))

    if dry:
        print('  （预演结束，未执行任何关闭动作）')
        return 0

    print('-' * 52)
    ok = 0
    done = set()
    for kind, label, pid in tg:
        if pid in done:
            continue
        if kill(pid):
            done.add(pid)
            ok += 1
            print('  ✓ 已关闭 %-7s %s' % (pid, label))
        else:
            print('  · %-7s 已不在（%s）' % (pid, label))
    time.sleep(2)

    # 自检
    print('-' * 52)
    print('  自检（还占着端口的）:')
    left = 0
    for m in (['ai', 'qq'] if mode == 'all' else [mode]):
        for port, label in PORTS.get(m, []):
            pids = port_pids(port)
            if pids:
                left += 1
                print('    ⚠ %-5d %-22s 仍在监听 (pid %s)' % (port, label, ','.join(pids)))
    if not left:
        print('    ✓ 所有相关端口都已释放')
    print('=== 完成：关掉 %d 个进程（共 %d 条目标） ===' % (ok, len(tg)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
