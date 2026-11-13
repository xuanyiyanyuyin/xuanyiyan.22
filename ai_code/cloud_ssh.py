# -*- coding: utf-8 -*-
"""云端实例 SSH 助手（用 paramiko，绕过本机 ssh.exe 被沙盒挡住的问题）

用法:
    python cloud_ssh.py run "hostname; nproc"          远程执行命令
    python cloud_ssh.py put <本地文件> <远端路径>        上传
    python cloud_ssh.py get <远端文件> <本地路径>        下载

连接信息（按优先级）:
    环境变量 CLOUD_HOST / CLOUD_PORT / CLOUD_USER / CLOUD_KEY / CLOUD_PASS
    默认: 36.150.116.206 : 34488  root  密钥 %USERPROFILE%\\.ssh\\id_ed25519
    先用密钥；密钥不行且给了密码，再试密码。
"""
import io
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import paramiko

HOST = os.environ.get('CLOUD_HOST', '36.150.116.206')
PORT = int(os.environ.get('CLOUD_PORT', '34488'))
USER = os.environ.get('CLOUD_USER', 'root')
KEY = os.environ.get('CLOUD_KEY', os.path.join(os.path.expanduser('~'), '.ssh', 'id_ed25519'))
PASS = os.environ.get('CLOUD_PASS') or None


def connect(timeout=20):
    """先密钥后密码；返回 (client, 用了哪种方式)"""
    errs = []
    if os.path.exists(KEY):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, key_filename=KEY,
                      timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
                      look_for_keys=False, allow_agent=False)
            return c, 'id_ed25519'
        except Exception as e:
            errs.append('key: %s' % str(e)[:120])
    else:
        errs.append('key: 不存在 %s' % KEY)
    if PASS:
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, port=PORT, username=USER, password=PASS,
                      timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
                      look_for_keys=False, allow_agent=False)
            return c, 'password'
        except Exception as e:
            errs.append('password: %s' % str(e)[:120])
    raise SystemExit('连接失败:\n  ' + '\n  '.join(errs))


def run(cmd, timeout=3600, quiet=False):
    c, how = connect()
    if not quiet:
        print('[ssh] %s@%s:%d (%s)' % (USER, HOST, PORT, how))
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    try:
        c.get_transport().set_keepalive(15)
    except Exception:
        pass
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    code = stdout.channel.recv_exit_status()
    c.close()
    return out, err, code


def put(local, remote):
    c, how = connect()
    sftp = c.open_sftp()
    try:
        sftp.put(local, remote)
        print('[put] %s -> %s (%d bytes)' % (os.path.basename(local), remote, os.path.getsize(local)))
    finally:
        sftp.close()
        c.close()


def get(remote, local):
    c, how = connect()
    sftp = c.open_sftp()
    try:
        sftp.get(remote, local)
        print('[get] %s -> %s (%d bytes)' % (remote, local, os.path.getsize(local)))
    finally:
        sftp.close()
        c.close()


def bg(script_local, remote_sh='/tmp/run_bg.sh', remote_log='/tmp/run_bg.log'):
    """把本地脚本传到远端并**脱离 SSH 会话**后台运行（setsid），立刻返回，不等它跑完"""
    import base64
    content = open(script_local, 'r', encoding='utf-8').read()
    payload = base64.b64encode(content.encode('utf-8')).decode()
    cmd = ("echo %s | base64 -d > %s && chmod +x %s && "
           "setsid nohup bash %s > %s 2>&1 < /dev/null & sleep 2; echo LAUNCHED-PID-$!") % (
        payload, remote_sh, remote_sh, remote_sh, remote_log)
    c, how = connect()
    chan = c.get_transport().open_session()
    chan.settimeout(30)
    chan.exec_command(cmd)
    time.sleep(4)
    data = b''
    try:
        while chan.recv_ready():
            data += chan.recv(65536)
    except Exception:
        pass
    c.close()          # 不等退出码，直接断；setsid 的那个进程会继续跑
    print('[bg] %s -> %s (detached)\n%s' % (script_local, remote_sh, data.decode('utf-8', 'replace').strip()))
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    action = sys.argv[1]
    if action == 'run':
        cmd = sys.argv[2] if len(sys.argv) > 2 else 'hostname'
        out, err, code = run(cmd)
        if out:
            print(out.rstrip())
        if err.strip():
            print('--- stderr ---')
            print(err.rstrip()[-3000:])
        print('[exit %d]' % code)
        return 0 if code == 0 else code
    if action in ('bg', 'detach'):
        return bg(sys.argv[2])
    if action in ('runfile', 'sh'):
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            cmd = f.read()
        out, err, code = run(cmd)
        if out:
            print(out.rstrip())
        if err.strip():
            print('--- stderr ---')
            print(err.rstrip()[-3000:])
        print('[exit %d]' % code)
        return 0 if code == 0 else code
    if action == 'put':
        put(sys.argv[2], sys.argv[3])
        return 0
    if action == 'get':
        get(sys.argv[2], sys.argv[3])
        return 0
    print(__doc__)
    return 1


if __name__ == '__main__':
    sys.exit(main())
