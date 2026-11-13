"""Qwen3.6-35B-A3B 服务 — 启动 llama-server + 联网代理, 供引擎管理器/CherryStudio 接入
CherryStudio 连 http://127.0.0.1:8081/v1
"""
import subprocess, time, os, sys, socket

LLAMACPP = r'D:\ai\llamacpp'
MODEL = r'D:/ai/models/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-AD-Q4_K_M.gguf'
PY = sys.executable

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port)); return False
        except OSError:
            return True

def stop():
    """停 llama-server + 代理"""
    import psutil
    for p in psutil.process_iter(['pid', 'name']):
        try:
            conns = p.connections()
            if any(c.laddr.port in (8080, 8081) for c in conns):
                p.terminate()
        except Exception:
            pass

def serve():
    stop()  # 清理残留
    # 启动 llama-server (attention GPU + MoE CPU, 262K 上下文)
    subprocess.Popen(
        [os.path.join(LLAMACPP, 'llama-server.exe'), '-m', MODEL,
         '-ngl', '999', '-cmoe', '-b', '512', '-ub', '256', '-np', '1', '-c', '32768', '-ctk', 'q8_0', '-ctv', 'q8_0',
         '-fa', 'on', '-t', '14', '--host', '127.0.0.1', '--port', '8080'],
        cwd=LLAMACPP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    # 等 llama-server 就绪
    for _ in range(120):
        if port_in_use(8080):
            break
        time.sleep(1)
    # 启动联网代理
    subprocess.Popen([PY, os.path.join(LLAMACPP, 'websearch_proxy.py')],
                     cwd=LLAMACPP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Qwen3.6 服务就绪 http://127.0.0.1:8081/v1', flush=True)
    # 保持常驻
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop()

if __name__ == '__main__':
    serve()
