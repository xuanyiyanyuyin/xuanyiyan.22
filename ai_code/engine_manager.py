"""
AI 引擎统一管理器 — 多模型互斥调度
架构: 所有引擎共享同一GPU/RAM, 一次只能运行一个
  - V4-Flash 对话  (v4flash_server.py, 常驻服务, 8123)
  - Qwen3.6 快速聊天 (qwen36_server.py, 常驻服务, 8081)
  - FLUX 生图  (gen_image.py, 任务型)
  - LTX 生视频 (gen_video.py, 任务型)
  - TTS/ASR   (gen_audio.py, 任务型)
  - VL 视觉   (vl_vqa.py, 任务型)
面板: http://127.0.0.1:8321
用法: D:\\ai\\python\\python.exe engine_manager.py
"""
import http.server, socketserver, json, os, sys, subprocess, threading, time, socket, signal
from pathlib import Path

BASE = Path(r'D:\ai')
PORT = 8321
PY = r'D:\ai\python\python.exe'

# ═══ 引擎注册表 ═══
# type: service=常驻HTTP, task=一次性任务
ENGINES = {
    'v4flash': {
        'name': 'V4-Flash 对话 (千文)',
        'type': 'service',
        'script': 'v4flash_chunked_server.py',   # 分块引擎（在线softmax + 流式prefill + 500专家）
        'py': r'D:\ai\python_ltx25\Scripts\python.exe',
        'port': 8123,
        'desc': 'DeepSeek-V4-Flash 156GB · FP4 · 在线分块 · 1M上下文 · 500专家',
        'ram_gb': 24, 'min_gpu_gb': 3, 'min_ram_gb': 16,
        'color': '#e040fb',
    },
    'qwen36': {
        'name': 'Qwen3.6-35B 快速聊天',
        'type': 'service',
        'script': 'qwen36_server.py',
        'py': r'D:\ai\python_ltx25\Scripts\python.exe',
        'port': 8081,
        'extra_ports': [8080],
        'desc': '20-25 tok/s · 262K上下文 · 联网搜索 · CherryStudio接8081',
        'ram_gb': 25, 'min_gpu_gb': 4, 'min_ram_gb': 12,
        'color': '#00bcd4',
    },
    'flux': {
        'name': 'FLUX 文生图 (三层搬运)',
        'type': 'task',
        'func': 'flux_3tier.py:text2img_3tier',
        'desc': '已由 FLUX.2 取代',
        'obsolete': True,
        'ram_gb': 12, 'min_gpu_gb': 4, 'min_ram_gb': 6,
        'color': '#ff6b35',
    },
    'img2img': {
        'name': 'FLUX 图生图/修复',
        'type': 'task',
        'func': 'gen_image.py:img2img_refine',
        'obsolete': True,
        'desc': '底图重绘修复 · 768×1376 · 低strength修细节保结构 · 不含IP-Adapter不OOM',
        'ram_gb': 22, 'min_gpu_gb': 4, 'min_ram_gb': 16,
        'color': '#7c4dff',
    },
    'flux3tier': {
        'name': 'FLUX 三层搬运',
        'type': 'task',
        'func': 'flux_3tier.py:img2img_3tier',
        'desc': '已由 FLUX.2 取代',
        'obsolete': True,
        'ram_gb': 22, 'min_gpu_gb': 4, 'min_ram_gb': 16,
        'color': '#00e5ff',
    },
    'flux2': {
        'name': 'FLUX.2 文生图/图生图',
        'type': 'task',
        'func': 'flux2_3tier.py:text2img_f2',
        'desc': '文本或参考图→图 · 多参考(≤6张) · 纯流式 · 官方对齐 · 512快速/2560质量',
        'ram_gb': 24, 'min_gpu_gb': 4, 'min_ram_gb': 12,
        'color': '#ff6b35',
    },
    'f2auto': {
        'name': 'FLUX.2 一键出图 (先编码→再生图)',
        'type': 'task',
        'func': 'flux2_encode_gen.py:main',
        'desc': '换新提示词只点这个: 自动判断 embeds 缓存 → 缺了先独立进程编码 → 立刻生图 · 512快/1024官方best',
        'ram_gb': 26, 'min_gpu_gb': 4, 'min_ram_gb': 12,
        'color': '#ff6b35',
    },
    'f2enc': {
        'name': 'FLUX.2 提示词编码 (生成 embeds 缓存)',
        'type': 'task',
        'func': 'flux2_encode.py:encode',
        'desc': '只编码不生成: 为提示词生成 embeds 缓存(独立进程, 45G text_encoder 逐层流式)',
        'ram_gb': 24, 'min_gpu_gb': 4, 'min_ram_gb': 12,
        'color': '#ff9d6b',
    },
    'ltx': {
        'name': 'LTX-13B 生视频',
        'type': 'task',
        'func': 'gen_video.py:generate_video',
        'desc': '2560×1408 原生2K · 17帧 · 3-4分钟',
        'ram_gb': 26, 'min_gpu_gb': 4, 'min_ram_gb': 20,
        'color': '#ff9d6b',
    },
    'i2v': {
        'name': 'LTX-2.5 图生视频',
        'type': 'task',
        'func': 'ltx25_e2e_i2v.py:main',
        'py': r'D:\ai\python_ltx25\Scripts\python.exe',
        'desc': '2560×1408 原生2K · 50步 · 图片首帧→视频 · 自动化出mp4',
        'ram_gb': 40, 'min_gpu_gb': 4, 'min_ram_gb': 20,
        'color': '#ff9d6b',
    },
    'audio': {
        'name': 'Qwen3 TTS/ASR',
        'type': 'task',
        'func': 'gen_audio.py:main',
        'desc': '语音合成 + 识别 · 双模型闭环',
        'ram_gb': 8, 'min_gpu_gb': 2, 'min_ram_gb': 4,
        'color': '#58a6ff',
    },
    'vl': {
        'name': 'Qwen3-VL-30B 视觉',
        'type': 'task',
        'func': 'vl_vqa.py:ask_image',
        'desc': '看图问答 · 官方推理 · 5分钟/次',
        'ram_gb': 25, 'min_gpu_gb': 4, 'min_ram_gb': 10,
        'color': '#3fb950',
    },
}

# ═══ 状态管理 ═══
class EngineState:
    def __init__(self):
        self.services = {}  # name -> Popen
        self.task_lock = threading.Lock()
        self.task_status = {}  # name -> {status, result, error}

state = EngineState()

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except OSError:
            return True

def is_service_running(name):
    eng = ENGINES[name]
    if eng['type'] != 'service':
        return False
    return port_in_use(eng['port']) or name in state.services and state.services[name].poll() is None

def stop_service(name):
    """停止常驻服务"""
    eng = ENGINES[name]
    if name in state.services:
        proc = state.services.pop(name)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    # 端口占用则杀进程 (含 extra_ports, 如 qwen36 的 llama-server 8080)
    import psutil
    ports = [eng['port']] + eng.get('extra_ports', [])
    for p in psutil.process_iter(['pid', 'name']):
        try:
            conns = p.connections()
            if any(c.laddr.port in ports for c in conns):
                p.terminate()
        except Exception:
            pass
    return True

def stop_all_services():
    """旧版: 只停 engine_manager 启动的服务。新版统一走 kill_all_engines"""
    kill_all_engines()

# ═══ 全局互斥: 只杀推理引擎进程 ═══
# 关键: 不碰 QQ 链路(NapCat/qq_bot/websearch_proxy) — 上次把 NapCat 的 3000 当引擎端口杀了, 导致 QQ 断线
ALL_ENGINE_PORTS = {8123, 8080, 8081}  # 推理引擎: v4flash(8123)/llama(8080)/qwen36代理(8081)
# 互斥识别: 常驻服务脚本 + 任务引擎代码特征 (subprocess 用 python -u -c "import <模块>...")
# 任务型也列入, 否则 kill_all_engines 杀不掉 gen_image/flux2 等一次性引擎, 互斥失效
ENGINE_SCRIPT_MARKS = ('v4flash', 'qwen36_server',
                       'flux2_3tier', 'flux_3tier', 'gen_image', 'gen_video', 'gen_audio',
                       'ltx25_e2e_i2v', 'vl_vqa', 'flux2_encode')  # 常驻服务 + 任务引擎模块名

# QQ 链路进程名（napcat框架 + qqbot + websearch_proxy + 本管理器），一律不杀
_QQ_LINK_NAMES = ('qq.exe', 'qqex.exe', 'napcatwinbootmain.exe', 'napcatbootmain.exe', 'napcat.exe')

def _engine_running_now():
    """[2026-10-03] 是否已有引擎脚本在跑（跑了就绝不清理，保护运行中的任务）"""
    try:
        import psutil
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = ' '.join(p.info['cmdline'] or [])
                if any(m in cmd for m in ENGINE_SCRIPT_MARKS) and p.info['pid'] != os.getpid():
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def kill_all_engines():
    """杀掉所有推理引擎进程（含手动启动的），保证互斥。
    [2026-10-03 主人令] 已有引擎在跑 ⇒ 直接跳过：绝不在任务运行中处决它
    （被处决在 CUDA 上下文里会留下驱动侧挂账，只有重启系统能清）。
    排除: engine_manager自身、websearch_proxy、qq_bot、NapCat(QQ.exe/QQEX/NapCatWinBootMain)"""
    if _engine_running_now():
        print('[互斥] 检测到引擎正在运行 → 跳过清理（不处决运行中的任务）', flush=True)
        return []
    import psutil
    killed = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pid = p.info['pid']
            cmd = ' '.join(p.info['cmdline'] or []).lower()
            name = (p.info['name'] or '').lower()
            # 排除: 自己 / QQ链路(websearch_proxy/qq_bot/napcat框架/QQ.exe)
            if 'engine_manager.py' in cmd:
                continue
            if 'websearch_proxy' in cmd or 'qq_bot.py' in cmd or 'napcat' in cmd:
                continue
            if name in _QQ_LINK_NAMES:
                continue
            # 判断是否推理引擎进程
            is_engine = False
            try:
                conns = p.connections()
                ports = [c.laddr.port for c in conns if c.status == 'LISTEN']
                if any(pt in ALL_ENGINE_PORTS for pt in ports):
                    is_engine = True
            except Exception:
                pass
            if any(m in cmd for m in ENGINE_SCRIPT_MARKS):
                is_engine = True
            if is_engine:
                # [2026-10-03 主人令] 优雅停优先：先终止，给它时间自己释放 CUDA/显存（硬杀会留系统侧挂账）
                try:
                    p.terminate()
                except Exception:
                    pass
                killed.append(pid)
        except Exception:
            pass
    # 确保被杀的进程退出
    # [2026-10-03] 优雅停：先等 5 秒看它自己退不退，仍在才强杀（硬杀会留系统侧挂账）
    time.sleep(5)
    import psutil as _ps
    for pid in killed:
        try:
            if _ps.pid_exists(pid):
                os.kill(pid, signal.SIGKILL if os.name != 'nt' else 9)
        except Exception:
            pass
    time.sleep(1.0)  # 等进程释放 GPU/RAM
    if killed:
        print(f'[互斥] 已停止 {len(killed)} 个引擎进程: {killed}', flush=True)
    return killed

# ═══ 资源预检查 (GPU 显存 + RAM, 避免 OOM) ═══
def gpu_free_gb():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().split('\n')[0]) / 1024.0
    except Exception:
        return -1.0  # 未知

def ram_free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return -1.0

def ensure_resources(min_gpu_gb, min_ram_gb):
    """启动引擎前检查 GPU/RAM 余量，不足则拒绝"""
    free_gpu = gpu_free_gb()
    free_ram = ram_free_gb()
    if free_gpu >= 0 and free_gpu < min_gpu_gb:
        raise RuntimeError(f'GPU 显存不足: 空闲 {free_gpu:.1f}GB < 需要 {min_gpu_gb}GB。请先停止其他引擎')
    if min_ram_gb and free_ram >= 0 and free_ram < min_ram_gb:
        raise RuntimeError(f'内存不足: 空闲 {free_ram:.1f}GB < 需要 {min_ram_gb}GB')
    return free_gpu, free_ram

def start_service(name):
    """启动常驻服务 — 全局互斥 + 资源预检查"""
    eng = ENGINES[name]
    # 有任务在跑(生图/视频/看图) → 拒绝启动服务, 避免抢占GPU打断任务
    running_tasks = [n for n, s in state.task_status.items() if s.get('status') == 'running']
    if running_tasks:
        print(f'[互斥] 有任务在跑: {running_tasks}，拒绝启动服务 {name}（等任务完成再启）', flush=True)
        return False
    # 互斥: 杀所有引擎服务进程（含手动启动的; 不杀任务进程）
    kill_all_engines()
    # 资源预检查
    ensure_resources(eng.get('min_gpu_gb', 3), eng.get('min_ram_gb', 8))
    py = eng.get('py', PY)
    proc = subprocess.Popen(
        [py, '-u', str(BASE / eng['script'])],
        cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    state.services[name] = proc
    # 等待端口就绪
    for _ in range(120):
        if port_in_use(eng['port']):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(1)
    return port_in_use(eng['port'])

def run_task(name, params):
    """跑一次性任务 (生图/视频/音频/VL) — 全局互斥 + 资源预检查"""
    eng = ENGINES[name]
    with state.task_lock:
        # 互斥: 杀所有引擎进程（含手动启动的）
        kill_all_engines()
        # 资源预检查 (LTX 需要 26GB RAM!)
        ensure_resources(eng.get('min_gpu_gb', 3), eng.get('min_ram_gb', 8))
        state.task_status[name] = {'status': 'running', 'params': params}
        try:
            result = _exec_task(name, params)
            state.task_status[name] = {'status': 'done', 'result': result}
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.task_status[name] = {'status': 'error', 'error': str(e)}
            raise

def _exec_task(name, params):
    """执行任务型引擎, 通过 subprocess 调用独立脚本"""
    eng = ENGINES[name]
    if name == 'flux':
        prompt = params.get('prompt', 'a red apple on white table, photorealistic')
        steps = min(int(params.get('steps', 50)), 200)
        width = int(params.get('width', 2560))
        height = int(params.get('height', 1408))
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import flux_3tier as ft\n"
            f"fn=ft.text2img_3tier({prompt!r}, steps={steps}, width={width}, height={height}, resident_d=0)\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-u', str(BASE / 'mem_release_runner.py'), '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=2400)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'img2img':
        img = params.get('image', '')
        prompt = params.get('prompt', '高清重绘, 保持人物一致')
        if not img or not os.path.exists(img):
            raise RuntimeError(f'图片不存在: {img}')
        strength = float(params.get('strength', 0.35))
        steps = min(int(params.get('steps', 28)), 100)
        width = int(params.get('width', 768))
        height = int(params.get('height', 1376))
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import gen_image as gi\n"
            f"fn=gi.img2img_refine({prompt!r}, {img!r}, strength={strength}, steps={steps}, width={width}, height={height})\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-u', str(BASE / 'mem_release_runner.py'), '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt, 'strength': strength, 'steps': steps, 'size': f'{width}x{height}'}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'flux3tier':
        img = params.get('image', '')
        prompt = params.get('prompt', '高清重绘, 保持人物一致')
        if not img or not os.path.exists(img):
            raise RuntimeError(f'图片不存在: {img}')
        strength = float(params.get('strength', 0.35))
        steps = min(int(params.get('steps', 50)), 200)
        width = int(params.get('width', 768))
        height = int(params.get('height', 1376))
        rd = int(params.get('resident_d', 12))
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import flux_3tier as ft\n"
            f"fn=ft.img2img_3tier({prompt!r}, {img!r}, strength={strength}, steps={steps}, width={width}, height={height}, resident_d={rd})\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-u', str(BASE / 'mem_release_runner.py'), '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=3600)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt, 'strength': strength, 'steps': steps, 'resident_d': rd}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'ltx':
        prompt = params.get('prompt', 'a red apple rotating on white table')
        img = params.get('image', '')  # 图生视频: 首帧图片路径
        if img:
            # 图生视频: 传 image 参数给 gen_video.generate_video
            code = (
                "import sys; sys.path.insert(0,r'D:/ai')\n"
                f"import gen_video as gv\n"
                f"fn,size,n=gv.generate_video({prompt!r}, steps={min(int(params.get('steps',50)),200)}, frames=17, base_w=2560, base_h=1408, do_upscale=False, input_image={img!r})\n"
                "print('RESULT:'+fn)"
            )
        else:
            # 文生视频
            code = (
                "import sys; sys.path.insert(0,r'D:/ai')\n"
                f"import gen_video as gv\n"
                f"fn,size,n=gv.generate_video({prompt!r}, steps={min(int(params.get('steps',50)),200)}, frames=17, base_w=2560, base_h=1408, do_upscale=False)\n"
                "print('RESULT:'+fn)"
            )
        out = subprocess.run([PY, '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'video': line[7:], 'prompt': prompt}
        raise RuntimeError('no output: '+out.stdout[-500:])
    elif name == 'i2v':
        # LTX-2.5 图生视频: 图片首帧 → latent → VAE decode → mp4 (脚本内自动完成)
        img = params.get('image', '')
        prompt = params.get('prompt', 'a cute orange cat sitting on a wooden table')
        if not img or not os.path.exists(img):
            raise RuntimeError(f'图片不存在: {img}')
        steps = str(params.get('steps', 50))
        frames = str(params.get('frames', 9))
        w = str(params.get('width', 2560))
        h = str(params.get('height', 1408))
        py_ltx = r'D:\ai\python_ltx25\Scripts\python.exe'
        out = subprocess.run([py_ltx, '-u', str(BASE / 'ltx25_e2e_i2v.py'), img, prompt, steps, frames, w, h],
                             cwd=str(BASE), capture_output=True, text=True, timeout=7200)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        # 返回最新生成的 mp4
        import glob
        mp4s = sorted(glob.glob(r'D:\ai\output\ab_test\lat_i2v_*.mp4'), key=os.path.getmtime)
        mp4 = mp4s[-1] if mp4s else ''
        return {'video': mp4, 'prompt': prompt, 'steps': steps, 'size': f'{w}x{h}'}
    elif name == 'flux2':
        # FLUX.2 文生图 / 图生图(多参考), 走 flux2_3tier.text2img_f2 (纯流式, 读embeds缓存)
        prompt = params.get('prompt', 'a cute orange cat on white table, photorealistic')
        steps = min(int(params.get('steps', 4)), 100)   # [2026-10-03] 上限 60→100
        width = int(params.get('width', 512))
        height = int(params.get('height', 512))
        rd = int(params.get('resident_d', 0))
        seed = params.get('seed', None)
        images = params.get('images', '')   # 参考图路径, 逗号分隔, 最多6张
        embeds_file = params.get('embeds_file', '')  # 可选, 现成embeds缓存
        # 构造 python 调用片段
        img_arg = (f"[{', '.join(repr(p.strip()) for p in images.split(',') if p.strip())}]"
                   if images else 'None')
        emb_arg = (f"{embeds_file!r}" if embeds_file else 'None')
        seed_arg = ('None' if seed in (None, '', 'null') else f'int({seed!r})')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import flux2_3tier as f2\n"
            f"fn=f2.text2img_f2({prompt!r}, steps={steps}, width={width}, height={height}, "
            f"resident_d={rd}, seed={seed_arg}, images={img_arg}, embeds_file={emb_arg})\n"
            "print('RESULT:'+fn)"
        )
        out = subprocess.run([PY, '-u', str(BASE / 'mem_release_runner.py'), '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=7200)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1200:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt, 'steps': steps, 'size': f'{width}x{height}'}
        raise RuntimeError('no output: ' + out.stdout[-500:])
    elif name == 'f2auto':
        # FLUX.2 一键: 先编码(子进程 45G text_encoder 逐层流式) → 再生图(纯流式 transformer)
        # 为什么要分进程: 45G + 66G 不能同场塞 32G 内存 (见 FLUX2_升级记录.md)
        prompt = params.get('prompt', 'a cute orange cat')
        steps = min(max(int(params.get('steps', 4)), 1), 100)   # [2026-10-03] 上限 60→100（50 是收益上限）
        width = int(params.get('width', 512))
        height = int(params.get('height', 512))
        rd = int(params.get('resident_d', 0))
        images = params.get('images', '')
        cmd = [PY, '-u', str(BASE / 'mem_release_runner.py'), str(BASE / 'flux2_encode_gen.py'), prompt, str(steps), str(width), str(height), str(rd)]
        if images:
            cmd.append(images)
        out = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True, timeout=10800)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout or '')[-1500:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'image': line[7:], 'prompt': prompt, 'steps': steps, 'size': f'{width}x{height}'}
        raise RuntimeError('no output: ' + out.stdout[-600:])
    elif name == 'f2enc':
        # FLUX.2 只编码: 为提示词生成 embeds 缓存(独立进程)
        prompt = params.get('prompt', 'a cute orange cat')
        out = subprocess.run([PY, '-u', str(BASE / 'mem_release_runner.py'), str(BASE / 'flux2_encode.py'), prompt, '512'],
                             cwd=str(BASE), capture_output=True, text=True, timeout=7200)
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout or '')[-1500:])
        for line in out.stdout.splitlines():
            if line.startswith('EMBEDS:'):
                return {'embeds': line[7:], 'prompt': prompt}
        raise RuntimeError('no output: ' + out.stdout[-600:])
    elif name == 'audio':
        text = params.get('text', '你好, 我是本地部署的语音助手。')
        voice = params.get('voice', '温柔清晰的年轻女性声音')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            "import gen_audio\n"
            f"gen_audio.main.__globals__['__name__']='x'\n"
            "import runpy\n"
            f"runpy.run_path(r'D:/ai/gen_audio.py', run_name='__main__')"
        )
        # gen_audio 用 argv, 通过环境传参
        env = dict(os.environ)
        out = subprocess.run([PY, 'gen_audio.py', text, voice], cwd=str(BASE),
                            capture_output=True, text=True, timeout=1200, env=env)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        # 找最新音频
        audio_dir = BASE / 'output' / 'audio'
        files = sorted(audio_dir.glob('*.wav'), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            return {'audio': str(files[0])}
        return {'log': out.stdout[-500:]}
    elif name == 'vl':
        img = params.get('image', '')
        question = params.get('question', '请用中文详细描述这张图片的内容。')
        code = (
            "import sys; sys.path.insert(0,r'D:/ai')\n"
            f"import vl_vqa\n"
            f"ans=vl_vqa.ask_image({img!r}, {question!r})\n"
            "print('RESULT:'+ans)"
        )
        out = subprocess.run([PY, '-c', code], cwd=str(BASE), capture_output=True, text=True, timeout=1200)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-1000:])
        for line in out.stdout.splitlines():
            if line.startswith('RESULT:'):
                return {'answer': line[7:], 'image': img}
        raise RuntimeError('no output: '+out.stdout[-500:])
    raise ValueError(f'unknown engine {name}')

# ═══ HTTP 服务 ═══
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def _send(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self._cors(); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/', '/index.html', '/panel'):
            body = (BASE / 'engine_panel.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self._cors(); self.end_headers()
            self.wfile.write(body)
        elif self.path == '/api/engines':
            data = {}
            for name, eng in ENGINES.items():
                running = is_service_running(name) if eng['type']=='service' else state.task_status.get(name,{}).get('status')=='running'
                data[name] = {**eng, 'running': running,
                              'task': state.task_status.get(name, {})}
            self._send({'engines': data})
        elif self.path == '/api/status':
            self._send({'services': {n: is_service_running(n) for n in ENGINES if ENGINES[n]['type']=='service'},
                        'tasks': {n: s.get('status') for n,s in state.task_status.items()}})
        else:
            # 静态图片/视频/音频
            if self.path.startswith('/output/'):
                fp = BASE / self.path.lstrip('/').replace('/', os.sep)
                if fp.exists():
                    ext = fp.suffix.lower()
                    ctype = {'png':'image/png','jpg':'image/jpeg','gif':'image/gif','wav':'audio/wav','mp4':'video/mp4'}.get(ext,'application/octet-stream')
                    body = fp.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', ctype)
                    self.send_header('Content-Length', str(len(body)))
                    self._cors(); self.end_headers()
                    self.wfile.write(body)
                    return
                self._send({'error':'not found'},404)
            else:
                self._send({'error':'unknown'},404)
    def do_POST(self):
        ln = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(ln).decode('utf-8')) if ln else {}
        except:
            req = {}
        if self.path == '/api/start':
            name = req.get('name','')
            if name not in ENGINES or ENGINES[name]['type'] != 'service':
                self._send({'error':'bad engine'},400); return
            # 后台线程启动, 避免阻塞HTTP
            def _start():
                start_service(name)
            threading.Thread(target=_start, daemon=True).start()
            self._send({'ok': True, 'engine': name, 'starting': True})
        elif self.path == '/api/stop':
            name = req.get('name','')
            if name not in ENGINES:
                self._send({'error':'bad engine'},400); return
            stop_service(name)
            self._send({'ok': True, 'engine': name, 'running': False})
        elif self.path == '/api/stop_all':
            stop_all_services()
            self._send({'ok': True})
        elif self.path == '/api/task':
            name = req.get('name','')
            params = req.get('params', {})
            if name not in ENGINES or ENGINES[name]['type'] != 'task':
                self._send({'error':'bad engine'},400); return
            # 后台线程跑, 避免阻塞HTTP
            def _run():
                try:
                    result = run_task(name, params)
                    state.task_status[name] = {'status':'done','result':result}
                except Exception as e:
                    state.task_status[name] = {'status':'error','error':str(e)}
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._send({'ok': True, 'status': 'running', 'engine': name})
        else:
            self._send({'error':'unknown'},404)

class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def main():
    print('='*50)
    print('  AI 引擎统一管理器')
    print(f'  面板: http://127.0.0.1:{PORT}')
    print('  引擎互斥: 一次只能运行一个')
    print('='*50)
    # 确保 psutil 可用
    try:
        import psutil
    except ImportError:
        print('[warn] psutil not installed, stop_service may be limited')
    srv = ThreadedServer(('127.0.0.1', PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_all_services()



# ══ [2026-10-02] 残留任务引擎收割 ══
#   事故: 面板被杀/重启后任务引擎变孤儿, python.exe 占 24.9GB 工作集挂了十几小时,
#   系统可用内存只剩 1.4GB, "虚拟内存居高不下"。下次启动/每 5 分钟自动收割。
ENGINE_MARKS = ('flux2_3tier', 'flux2_encode_gen', 'flux2_encode', 'flux_3tier',
                'gen_image.py', 'gen_video.py', 'ltx_client')



def _alive_pids():
    """当前活着的 PID 集合（用于判断父进程是否还在 = 是否真孤儿）"""
    try:
        import psutil
        return {q.pid for q in psutil.process_iter(['pid'])}
    except Exception:
        return set()

def reap_orphan_engines(verbose=False):
    """杀掉「不是本面板子进程」的残留任务引擎; 返回 [(pid, rss_mb), ...]"""
    killed = []
    try:
        import psutil
        me = os.getpid()
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid', 'create_time']):
            try:
                if 'python' not in (p.info['name'] or '').lower():
                    continue
                cmd = ' '.join(p.info['cmdline'] or [])
                if not any(m in cmd for m in ENGINE_MARKS):
                    continue
                if p.info['pid'] == me or p.info.get('ppid') == me:
                    continue
                # [2026-10-02 修正] 只杀【真孤儿】：父进程已不存在。
                #   原来按"父进程不是面板就杀"⇒ 误杀了手动/脚本启动的合法引擎
                #   （实测：pwsh 启动的 flux2_encode_gen 跑到一半被收割, exit code 1、无报错）。
                ppid = p.info.get('ppid') or 0
                if ppid and ppid in _alive_pids():
                    continue
                if ppid == 0:
                    continue
                # [2026-10-02 主人令] 安全阀：只收割存活 >30 分钟的残留。
                #   实测：面板经 cmd/start 启动的 bf16 生图，父进程(启动壳)先退出 ⇒ 被误当孤儿杀掉
                #   （日志实证 [reaper] 收割残留引擎: PID2376(10339MB)）。运行中的新进程绝不碰。
                try:
                    _age = time.time() - float(p.info.get('create_time') or 0)
                except Exception:
                    _age = 0.0
                if _age < 1800:
                    continue
                try:
                    rss = p.memory_info().rss / 1048576
                except Exception:
                    rss = -1
                p.kill()
                killed.append((p.info['pid'], rss))
            except Exception:
                pass
    except Exception:
        pass
    if killed:
        print('[reaper] 收割残留引擎: ' + ', '.join('PID%d(%.0fMB)' % k for k in killed), flush=True)
    elif verbose:
        print('[reaper] 无残留引擎', flush=True)
    return killed


def _reaper_loop(interval=300):
    import time as _t
    while True:
        _t.sleep(interval)
        try:
            pass  # 收割器已撤
        except Exception:
            pass


if __name__ == '__main__':
    try:
        pass  # [2026-10-03 主人令] 面板已有互斥，撤掉收割器（它会误杀运行中的引擎）          # [2026-10-02] 启动先清一遍孤儿
        pass  # [2026-10-03 主人令] 不再启动收割线程（互斥已足够）
    except Exception as _e:
        print('[reaper] 启动失败: %s' % str(_e)[:120], flush=True)
    main()
