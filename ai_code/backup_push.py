#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[2026-10-02] 一键备份：把本地代码同步到 GitHub 私有仓库 xuanyiyanyuyin/ai-stack

用法（聊天栈 python）：
    python backup_push.py            增量备份并推送
    python backup_push.py --dry      只看会改什么，不推

做四件事：
  1. 重新收集代码（排除模型/输出/二进制/日志）
  2. 把真密钥替换成占位符（真值留在本机 _密钥备份_*.txt，不进仓库）
  3. git add/commit
  4. push 到 origin main（走 ~/.ssh/config 里的 ssh.github.com:443）
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

GIT = r'C:\Program Files\Git\cmd\git.exe'
STAGE = r'D:\ai\_backup_ai_stack'
SRC = r'D:\ai'
LOG = r'D:\ai\_backup_ai_stack\_last_backup.log'
DRY = '--dry' in sys.argv

EXCLUDE_DIRS = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', 'models', 'output',
                '_backup_ai_stack', 'site-packages', '.pylibs', 'temp', 'tmp', 'cache_data'}
EXCLUDE_EXT = {'.dll', '.zip', '.exe', '.pyd', '.so', '.bin', '.gguf', '.safetensors', '.pt',
               '.mp4', '.mp3', '.wav', '.png', '.jpg', '.jpeg', '.webp', '.ico', '.onnx', '.log'}
MAX_SIZE = 2 * 1024 * 1024
PLAN = [(r'D:\ai\llamacpp', 'llamacpp'), (r'D:\ai\LingBot_脚本', 'LingBot_脚本'),
        (r'D:\ai\frontend', 'frontend'),
        (r'D:\ai\经验库_训练侧补充', '经验库_训练侧补充'),
        (r'D:\ai\经验库', '经验库'),
        (r'D:\ai\_v4flash_src', '_v4flash_src'),
        (r'D:\ai\LingBot_实例产物_20260923_单卡冒烟', 'LingBot_单卡冒烟产物'),
        (r'D:\ai\LingBot_初赛提交材料', 'LingBot_初赛提交材料'),
        # ── 下面这些是大目录：只收代码/配置类小文件，权重与产物自动被过滤掉 ──
        (r'D:\ai\_ltx25_src', '魔改引擎_LTX25_src'),
        (r'D:\ai\LingBot_8卡_产物_20260926', '比赛_8卡产物_代码与配置'),
        (r'D:\ai\LingBot_实例产物_20260923_单卡冒烟', '比赛_单卡产物_代码与配置'),
        (r'D:\ai\models', '模型配置(无权重)'),
        (r'D:\ai\ComfyUI\user', 'ComfyUI_用户工作流'),
        (r'D:\ai\nka', 'nka')]


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    print(line)
    try:
        open(LOG, 'a', encoding='utf-8').write(line + '\n')
    except Exception:
        pass


def copy_tree(src, dst):
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel = os.path.relpath(dirpath, src)
        target = os.path.join(dst, rel) if rel != '.' else dst
        os.makedirs(target, exist_ok=True)
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in EXCLUDE_EXT:
                continue
            fp = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(fp) > MAX_SIZE:
                    continue
                shutil.copy2(fp, os.path.join(target, fn))
                n += 1
            except Exception:
                pass
    return n


def collect():
    if os.path.exists(STAGE):
        for name in os.listdir(STAGE):
            if name != '.git':
                p = os.path.join(STAGE, name)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    os.makedirs(STAGE, exist_ok=True)
    total = 0
    for src, dst in PLAN:
        if os.path.isdir(src):
            total += copy_tree(src, os.path.join(STAGE, dst))
    os.makedirs(os.path.join(STAGE, 'root_scripts'), exist_ok=True)
    for fn in os.listdir(SRC):
        fp = os.path.join(SRC, fn)
        if not os.path.isfile(fp) or fn == 'qq_official.json':
            continue
        if os.path.splitext(fn)[1].lower() not in ('.py', '.md', '.txt', '.bat', '.json', '.sh', '.yaml', '.yml'):
            continue
        if os.path.getsize(fp) > MAX_SIZE:
            continue
        shutil.copy2(fp, os.path.join(STAGE, 'root_scripts', fn))
        total += 1
    # [2026-10-02] 每次备份都重建这两个文件（原来只在首次生成 ⇒ 第二次备份被清掉）
    try:
        with open(os.path.join(STAGE, '.gitignore'), 'w', encoding='utf-8') as fh:
            fh.write('# 秘密与本地状态，一律不进仓库\n'
                     'qq_official.json\n*_secrets*.txt\n*_密钥备份*.txt\n'
                     '__pycache__/\n*.pyc\n*.log\n*.dll\n*.zip\n*.exe\n*.gguf\n*.safetensors\n*.pt\n'
                     'assets/\nnode_modules/\n.venv/\n')
        with open(os.path.join(STAGE, 'qq_official.example.json'), 'w', encoding='utf-8') as fh:
            json.dump({'app_id': '你的AppID', 'app_secret': '你的AppSecret', 'owner_map': {}, 'names': {},
                       'media_token': '随机串（网关自动生成）',
                       '_note': '真实文件是 D:\\ai\\qq_official.json，不进仓库；复制本模板改名后填自己的值'},
                      fh, ensure_ascii=False, indent=2)
    except Exception as e:
        log('模板重建失败: %s' % str(e)[:80])
    log('收集完成：%d 个文件' % total)
    return total


def sanitize():
    pat = re.compile(r'sk-[A-Za-z0-9]{24,}')
    n = 0
    for dirpath, _, filenames in os.walk(STAGE):
        if '.git' in dirpath:
            continue
        for fn in filenames:
            if not fn.lower().endswith(('.py', '.json', '.md', '.txt', '.sh', '.bat', '.yaml', '.yml')):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                t = open(fp, encoding='utf-8', errors='replace').read()
            except Exception:
                continue
            o = t
            for m in set(pat.findall(t)):
                if 'PUT_YOUR' in m or 'YOUR_KEY' in m:
                    continue
                t = t.replace(m, 'sk-你的KEY')
            if t != o:
                open(fp, 'w', encoding='utf-8').write(t)
                n += 1
    log('密钥占位符替换：%d 个文件' % n)


def git(*args, timeout=900):
    p = subprocess.run([GIT] + list(args), cwd=STAGE, capture_output=True, timeout=timeout,
                       env=dict(os.environ, GCM_INTERACTIVE='never', GIT_TERMINAL_PROMPT='0'))
    return p.returncode, ((p.stdout or b'').decode('utf-8', 'replace') + (p.stderr or b'').decode('utf-8', 'replace'))


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    log('=== 备份开始%s ===' % ('（dry-run）' if DRY else ''))
    collect()
    sanitize()
    rc, out = git('add', '-A')
    rc, out = git('status', '--short')
    changed = len([l for l in out.splitlines() if l.strip()])
    log('变更条目：%d' % changed)
    if DRY:
        for l in out.splitlines()[:20]:
            log('  ' + l[:120])
        sys.exit(0)
    if changed == 0:
        log('没有变化，无需推送')
        sys.exit(0)
    rc, out = git('-c', 'core.safecrlf=false', 'commit', '-m',
                  '同步备份 %s（密钥已占位符化）' % time.strftime('%Y-%m-%d %H:%M'))
    log('commit rc=%s' % rc)
    rc, out = git('push', 'origin', 'main')
    log('push rc=%s' % rc)
    for l in out.strip().splitlines()[-3:]:
        log('  ' + l[:140])
    log('=== 备份结束 ===')
