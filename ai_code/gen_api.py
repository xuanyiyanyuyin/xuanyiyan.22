# -*- coding: utf-8 -*-
"""统一生成 API：生图（FLUX.2-klein-4B）+ 生视频（LTX-2.5 GGUF / ComfyUI）

默认端口 8330，绑定 0.0.0.0（局域网内其它服务/机器人可直接调用）。
设了环境变量 GEN_API_KEY 就要求鉴权（请求头 X-API-Key 或 Authorization: Bearer）。

接口一览
--------
GET  /health                     健康检查 + 引擎状态
POST /v1/image                   生图（同步，约 25 秒）
     {"prompt": "...", "width":864, "height":1536, "steps":28, "seed":1234}   # 省略 width/height 即默认竖版 864x1536
POST /v1/images/generations      OpenAI 风格生图（同步）
     {"prompt": "...", "size":"864x1536", "n":1}   # 横版用 "1536x864"
POST /v1/video                   生视频（异步，返回 job_id；512x320/49帧约 2.5 分钟）
     {"prompt": "...", "width":768, "height":512, "frames":97, "upscale":false, "seed":42}
GET  /v1/video/<job_id>          查询视频任务
GET  /v1/jobs                    列出任务
POST /v1/free                    释放显存（卸载 klein / 让 ComfyUI 卸模型）
GET  /files/<filename>           取生成的文件（图片/视频）

命令行自测
--------
python gen_api.py --selftest
"""
import json
import os
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── 路径与常量 ──────────────────────────────────────────────────────────────
KLEIN_DIR = r'D:\ai\models\FLUX.2-klein-4B'
OUT_DIR = r'D:\ai\output\gen_api'
COMFY_DIR = r'D:\ai\ComfyUI'
COMFY_OUT = os.path.join(COMFY_DIR, 'output')
PORT = int(os.environ.get('GEN_API_PORT', '8330'))
API_KEY = os.environ.get('GEN_API_KEY', '').strip()
IDLE_UNLOAD_SEC = int(os.environ.get('GEN_API_IDLE_UNLOAD', '120'))   # 空闲兜底卸载
HOLD_SEC = int(os.environ.get('GEN_API_HOLD_SEC', '0'))               # 0 = 生完立刻卸载（省内存，重载很快）
BUSY = {'n': 0}                                                       # 正在生成的请求数（卸载要看它）

os.makedirs(OUT_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:            # 本机 python 是嵌入式布局，脚本目录不会自动进 sys.path
    sys.path.insert(0, BASE_DIR)
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

GPU_LOCK = threading.Lock()          # 同一时刻只跑一个重活（生图/生视频）
S = {
    'klein': None,                   # diffusers pipeline（惰性加载）
    'klein_last_use': 0.0,
    'jobs': {},                      # job_id -> {status, params, result, ...}
    'job_seq': 0,
    'started': time.time(),
}


def log(msg):
    line = time.strftime('%H:%M:%S ') + str(msg)
    print(line, flush=True)
    try:
        with open(os.path.join(os.environ.get('TEMP', r'C:\Windows\Temp'), 'gen_api.log'), 'a',
                  encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ── klein 生图 ──────────────────────────────────────────────────────────────
def load_klein():
    if S['klein'] is not None:
        return S['klein']
    import torch
    from diffusers import Flux2KleinPipeline
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(KLEIN_DIR, torch_dtype=torch.bfloat16, local_files_only=True)
    # [2026-10-15 主人令：只放显存，不许走共享显存]
    # 依据：WDDM 共享显存 = 权重走 PCIe，扩散每图数千 kernel ⇒ 慢 10-100x（H3 同一坑：显存贴顶 27-37s/层 vs 有富余 3.3s/层）
    mode = 'none'
    _free_gb = 0.0
    try:
        import torch as _t
        _free_b, _ = _t.cuda.mem_get_info()
        _free_gb = _free_b / 1024 ** 3
    except Exception:
        pass
    _vram_only = os.environ.get('GEN_VRAM_ONLY', '0') != '0'   # [2026-10-15] 默认关：显存不足时硬塞会溢出，确认腾够显存再开
    _vram_min = float(os.environ.get('GEN_VRAM_MIN_GB', '8.5'))
    if _vram_only and _free_gb >= _vram_min:
        try:
            # [2026-10-15 主人令：拆开放] 双流层全常驻 GPU；单流层前 K 层常驻，其余留 CPU 按需搬（H3 同款思路）
            _k = int(os.environ.get('GEN_SPLIT_SINGLE_ON_GPU', '12'))
            _moved = 0
            try:
                for _blk in getattr(pipe.transformer, 'transformer_blocks', []):
                    _blk.to('cuda'); _moved += 1
            except Exception as _e:
                log('双流层上卡失败: %s' % str(_e)[:80])
            _sb = list(getattr(pipe.transformer, 'single_transformer_blocks', []) or [])
            for _i, _blk in enumerate(_sb):
                _blk.to('cuda'); _moved += 1
            for _name in ('x_embedder', 'context_embedder', 'time_text_embed', 'norm_out', 'proj_out'):
                _mod = getattr(pipe.transformer, _name, None)
                if _mod is not None:
                    try: _mod.to('cuda')
                    except Exception: pass
            # [2026-10-15] 原生 group offload：块级分组 + CUDA 流重叠，设备记账由 diffusers 自己管
            try:
                from diffusers.hooks import apply_group_offloading
                _gb = int(os.environ.get('GEN_GROUP_BLOCKS', '5'))
                _place = os.environ.get('GEN_PLACEMENT', 'auto')
                if _place == 'auto':
                    _place = 'full' if _free_gb >= float(os.environ.get('GEN_FULL_MIN_GB', '10.5')) else 'group'
                log('放置策略: %s（加载时空闲显存 %.1fGB）' % (_place, _free_gb))
                if _place == 'full':
                    raise RuntimeError('skip-group-offload: 全放 GPU，不搬运')
                apply_group_offloading(pipe.transformer, onload_device=torch.device('cuda'),
                                       offload_device=torch.device('cpu'), offload_type='block_level',
                                       num_blocks_per_group=_gb, use_stream=True, record_stream=True)
                log('已启用 group offload（每组 %d 块 + CUDA 流重叠）' % _gb)
            except Exception as _e:
                log('group offload 失败，改为全放 GPU: %s' % str(_e)[:100])
            pipe.vae.to('cuda')
            log('分层放置: 双流+单流前 %d 层常驻 GPU，其余 %d 层按需搬' % (_k, max(0, len(_sb) - _k)))
            # [2026-10-15] 关键：text_encoder 留 CPU 后 diffusers 会把 self.device 判成 CPU，
            # 于是 latents/ids 建在 CPU、权重在 GPU ⇒ 设备不匹配。这里把执行设备钉成 cuda（仅本进程生效）
            try:
                _cls = type(pipe)
                _cls.device = property(lambda self: torch.device('cuda'))
                _cls._execution_device = property(lambda self: torch.device('cuda'))
                log('执行设备已钉为 cuda（text_encoder 仍留 CPU）')
            except Exception as _e:
                log('设备钉定失败: %s' % str(_e)[:80])
            try:
                pipe.text_encoder.to('cpu')
            except Exception:
                pass
            mode = 'vram_only(text_enc_cpu)'
            # [2026-10-15] 再省 1-2GB 峰值：VAE 解码挪 CPU + 打开 attention/vae slicing
            try:
                pipe.vae.enable_tiling()   # VAE 仅 0.3GB，留 GPU（挪 CPU 会与 cuda 的 latent 撞设备）
            except Exception:
                pass
            try:
                pipe.enable_attention_slicing('auto')
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        except Exception as e:
            log('vram_only 放置失败，回落 offload: %s' % str(e)[:120])
            mode = 'none'
    else:
        log('空闲显存 %.1fGB < 阈值 %.1fGB ⇒ 走 offload（会慢，建议先关占显存的桌面程序）' % (_free_gb, _vram_min))
    if mode == 'none':
        for name, fn in (('model_cpu_offload', pipe.enable_model_cpu_offload),
                         ('sequential_cpu_offload', pipe.enable_sequential_cpu_offload)):
            try:
                fn(); mode = name; break
            except Exception as e:
                log('offload %s 失败: %s' % (name, str(e)[:120]))
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    S['klein'] = pipe
    S['klein_mode'] = mode
    log('klein 加载完成 %.1fs，放置=%s（加载时空闲显存 %.1fGB）' % (time.time() - t0, mode, _free_gb))
    return pipe


def _mem_stats():
    """自身内存(RSS/私有/虚拟 MB) + 系统可用(MB)；拿不到返回 0"""
    try:
        import psutil
        m = psutil.Process().memory_info()
        return (m.rss / 1048576, (getattr(m, 'private', 0) or 0) / 1048576,
                (getattr(m, 'vms', 0) or 0) / 1048576, psutil.virtual_memory().available / 1048576)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def _trim_memory(tag='job'):
    # [2026-10-03 主人令·防泄漏] 任务结束顺带让 ComfyUI 释放模型/显存
    try:
        if os.environ.get('FREE_COMFY_AFTER_JOB', '1') == '1':
            import ltx_client
            ltx_client.free_comfy()
    except Exception:
        pass
    # [2026-10-03 主人令] 跑完即卸：任务结束立刻把管线/显存/提交还回去（原来只在内存吃紧时才卸）
    try:
        if os.environ.get('GEN_API_UNLOAD_AFTER_JOB', '1') == '1':
            unload_klein('跑完即卸')
    except Exception:
        pass
    """[2026-10-01 主人令·内存泄露] 每次生成后做内存卫生：
    GC → 清 torch 显存缓存 → EmptyWorkingSet 把闲置页还给系统（任务管理器里"内存/虚拟内存"立刻回落）。
    连续生图/生视频时 Windows 提交量一路涨，根因就是这些缓存+堆碎片不会自己还。"""
    if os.environ.get('GEN_API_TRIM', '1') == '0':
        return
    try:
        import gc
        gc.collect(); gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass
    rss, priv, vms, avail = _mem_stats()
    log('[mem %s] RSS=%.0fMB 私有=%.0fMB 虚拟=%.0fMB | 系统可用=%.1fGB' % (tag, rss, priv, vms, avail / 1024))


def _free_comfy_if_tight(why=''):
    # [2026-10-02 主人令] 运行中绝不干预
    try:
        if any(j.get('status') == 'running' for j in (S.get('jobs') or {}).values()):
            return
    except Exception:
        pass
    """系统可用内存低于阈值时让 ComfyUI 卸模型（否则它的常驻模型会一直占着）"""
    try:
        rss, priv, vms, avail = _mem_stats()
        thr = float(os.environ.get('FREE_COMFY_RAM_GB', '6')) * 1024
        if avail and avail < thr:
            import ltx_client
            ltx_client.free_comfy()
            log('[mem] 可用内存 %.1fGB < %.1fGB → 让 ComfyUI 卸模型 %s' % (avail / 1024, thr / 1024, why))
            return True
    except Exception as e:
        log('[mem] 释放 ComfyUI 失败: %s' % str(e)[:120])
    return False


def _trim_jobs(keep=60):
    """任务表只留最近 N 条（原来无界增长：S['jobs'] 永不清理）"""
    try:
        jobs = S.get('jobs') or {}
        if len(jobs) > keep:
            for k in sorted(jobs, key=lambda x: jobs[x].get('created', 0))[:-keep]:
                jobs.pop(k, None)
    except Exception:
        pass


def _mem_guard(where='job'):
    # [2026-10-02 主人令] 运行中绝不干预：有任务在跑就直接返回
    try:
        if any(j.get('status') == 'running' for j in (S.get('jobs') or {}).values()):
            return
    except Exception:
        pass
    """[2026-10-01 主人令·内存泄露] 守门：
    私有内存 ≥ 软限（默认 6000MB）→ 卸载 klein + 回收 + 归还工作集；
    ≥ 硬限（默认 10000MB）且空闲 → 若 GEN_API_SELF_RESTART=1 则自我重启。
    背景：2026-10-01 20:51 系统日志记录 python.exe 涨到 64.7GB 虚拟内存 → 20:53 栈溢出崩溃、全栈服务挂掉。"""
    try:
        rss, priv, vms, avail = _mem_stats()
        soft = float(os.environ.get('GEN_API_MEM_SOFT_MB', '6000'))
        hard = float(os.environ.get('GEN_API_MEM_HARD_MB', '10000'))
        if priv <= 0:
            return
        if priv >= soft:
            unload_klein('内存守门 私有%.0fMB' % priv)
            _trim_memory('guard')
            rss2, priv2, _, _ = _mem_stats()
            log('[mem-guard] %s 私有 %.0fMB ≥ 软限 %.0fMB → 卸载+回收 → %.0fMB' % (where, priv, soft, priv2))
            priv = priv2
        if priv >= hard:
            log('[mem-guard-ALERT] %s 私有 %.0fMB ≥ 硬限 %.0fMB（若继续涨会重演 64GB 崩溃）' % (where, priv, hard))
            if BUSY['n'] == 0 and os.environ.get('GEN_API_SELF_RESTART', '0') == '1':
                log('[mem-guard] 空闲且已超硬限 → 自我重启')
                try:
                    import subprocess, sys
                    subprocess.Popen([sys.executable, '-u'] + sys.argv,
                                     cwd=os.path.dirname(os.path.abspath(__file__)),
                                     creationflags=0x00000200 | 0x08000000)
                except Exception as e:
                    log('[mem-guard] 自我重启失败: %s' % str(e)[:120])
                os._exit(0)
    except Exception as e:
        log('[mem-guard] 异常: %s' % str(e)[:150])


def unload_klein(reason='idle'):
    if S['klein'] is None:
        return False
    try:
        import gc
        import torch
        del S['klein']
        S['klein'] = None
        gc.collect()
        torch.cuda.empty_cache()
        log('klein 已卸载（%s）' % reason)
        return True
    except Exception as e:
        log('klein 卸载异常: %s' % str(e)[:150])
        return False


def idle_watchdog():
    while True:
        time.sleep(30)
        try:
            if S['klein'] is not None and IDLE_UNLOAD_SEC > 0 and BUSY['n'] == 0:
                if time.time() - S['klein_last_use'] > IDLE_UNLOAD_SEC:
                    unload_klein('空闲 %ds' % IDLE_UNLOAD_SEC)
        except Exception:
            pass


# [2026-09-30 主人令] 默认尺寸改为模型/API 自己的默认：竖版 864x1536；横版 1536x864
def gen_image(prompt, width=864, height=1536, steps=20, seed=None, out_name=None):
    import torch
    with GPU_LOCK:
        BUSY['n'] += 1
        try:
            # 生图前让 ComfyUI 让出显存
            try:
                import ltx_client
                ltx_client.free_comfy()
            except Exception:
                pass
            pipe = load_klein()
            S['klein_last_use'] = time.time()
            gen = torch.Generator(device='cpu')
            if seed is None:
                seed = int(time.time() * 1000) % (2 ** 31)
            gen.manual_seed(int(seed))
            t0 = time.time()
            # [2026-10-15] vram_only 模式：文本编码只在 CPU 跑一次，条件 embedding 再送 GPU 去噪
            # 这样 text_encoder(7.7GB) 不占显存，transformer+VAE 才能常驻 GPU、不碰共享内存
            if str(S.get('klein_mode') or '').startswith('vram_only'):
                with torch.no_grad():
                    _pe, _tids = pipe.encode_prompt(prompt=prompt, device='cpu', num_images_per_prompt=1,
                                                    max_sequence_length=512, text_encoder_out_layers=(9, 18, 27))
                img = pipe(prompt=None, prompt_embeds=_pe.to('cuda'), height=int(height), width=int(width),
                           num_inference_steps=int(steps), guidance_scale=1.0,
                           generator=gen, num_images_per_prompt=1).images[0]
            else:
                img = pipe(prompt=prompt, height=int(height), width=int(width),
                           num_inference_steps=int(steps), guidance_scale=1.0,
                           generator=gen, num_images_per_prompt=1).images[0]
            name = out_name or ('klein_%d_%d.png' % (int(time.time()), seed))
            path = os.path.join(OUT_DIR, name)
            img.save(path)
            elapsed = round(time.time() - t0, 1)
        finally:
            BUSY['n'] -= 1
            # 默认"用完即卸"：klein 常驻要占约 18GB 内存，32G 机器扛不住；重载只要几秒
            if HOLD_SEC > 0:
                S['klein_last_use'] = time.time()
            else:
                unload_klein('用完即卸（省内存）')
        return {'file': path, 'filename': name, 'seed': int(seed),
                'width': int(width), 'height': int(height), 'steps': int(steps),
                'elapsed': elapsed}


# ── LTX 生视频 ──────────────────────────────────────────────────────────────
def gen_upscale(path, scale=2):
    """超分：图片走 4x-UltraSharp（spandrel），视频走逐帧超分 + ffmpeg 重编码"""
    import upscale_lib
    with GPU_LOCK:
        BUSY['n'] += 1
        try:
            unload_klein('超分前让显存')
            try:
                import ltx_client
                ltx_client.free_comfy()
            except Exception:
                pass
            low = path.lower()
            if low.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                return upscale_lib.upscale_image(path, scale=scale)
            return upscale_lib.upscale_video(path, scale=scale)
        finally:
            BUSY['n'] -= 1


def gen_video_sync(prompt, width=768, height=512, frames=97, upscale=False, seed=42, image_path=None, sr=0,
                   cancel_check=None):
    import shutil
    import ltx_client
    with GPU_LOCK:
        if cancel_check and cancel_check():
            raise RuntimeError('cancelled')
        unload_klein('生视频前让显存')
        t0 = time.time()
        res = ltx_client.generate(prompt, width=width, height=height, frames=frames,
                                  upscale=upscale, seed=seed, prefix='video/gen_api_%d' % int(time.time()),
                                  image_path=image_path)
        info = res.get('file')
        if not info:
            raise RuntimeError('视频任务未产出文件：status=%s outputs=%s' % (
                res.get('status'), json.dumps(res.get('outputs', {}), ensure_ascii=False)[:300]))
        src = os.path.join(COMFY_OUT, info.get('subfolder', ''), info['filename'])
        dst_name = 'ltx_%d_%s' % (int(time.time()), os.path.basename(info['filename']))
        dst = os.path.join(OUT_DIR, dst_name)
        shutil.copy2(src, dst)
        if cancel_check and cancel_check():          # 生成完但被取消 → 超分不做了
            raise RuntimeError('cancelled')
        res = {'file': dst, 'filename': dst_name, 'source': src,
               'status': res.get('status'), 'mode': res.get('mode', 't2v'),
               'image': image_path or '', 'width': int(width), 'height': int(height),
               'frames': int(frames), 'upscale': bool(upscale), 'seed': int(seed),
               'elapsed': round(time.time() - t0, 1)}
        # 生成后超分（可选）：逐帧 4x-UltraSharp + ffmpeg 重编码，音轨保留
        if sr and int(sr) > 0:
            try:
                import upscale_lib
                r2 = upscale_lib.upscale_video(dst, scale=int(sr))
                res['sr'] = int(sr)
                res['sr_file'] = r2['file']
                res['sr_width'] = r2['width']
                res['sr_height'] = r2['height']
                res['file'] = r2['file']            # 交出去的用超分版
                res['filename'] = os.path.basename(r2['file'])
                res['elapsed'] = round(time.time() - t0, 1)
                res['sr_elapsed'] = r2['elapsed']
            except Exception as e:
                res['sr_error'] = str(e)[:200]
        return res


def _run_video_job(job_id, params):
    job = S['jobs'][job_id]
    if job.get('cancelled'):
        job['status'] = 'cancelled'
        return
    job['status'] = 'running'
    job['started'] = time.time()
    try:
        r = gen_video_sync(**params, cancel_check=lambda: bool(job.get('cancelled')))
        job['result'] = r
        job['status'] = 'done'
    except Exception as e:
        if 'cancelled' in str(e):
            job['status'] = 'cancelled'
            log('视频任务已取消 %s' % job_id)
        else:
            job['status'] = 'error'
            job['error'] = str(e)[:500]
            log('视频任务失败 %s: %s\n%s' % (job_id, str(e)[:200], traceback.format_exc()[-800:]))
    job['finished'] = time.time()
    _trim_jobs()                       # [2026-10-01] 任务表只留最近 60 条
    _trim_memory('video')              # [2026-10-01] 视频收尾回收（含逐帧超分留下的堆）
    _mem_guard('video')
    _free_comfy_if_tight('视频后')


def new_video_job(params):
    S['job_seq'] += 1
    job_id = 'v%d_%d' % (S['job_seq'], int(time.time()))
    S['jobs'][job_id] = {'job_id': job_id, 'type': 'video', 'params': params,
                         'status': 'queued', 'created': time.time()}
    threading.Thread(target=_run_video_job, args=(job_id, params), daemon=True).start()
    return job_id


# ── HTTP 层 ────────────────────────────────────────────────────────────────
def json_bytes(obj, code=200):
    return code, json.dumps(obj, ensure_ascii=False).encode('utf-8')


class Handler(BaseHTTPRequestHandler):
    server_version = 'GenAPI/1.0'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):      # 静音默认日志
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self):
        if not API_KEY:
            return True
        h = self.headers.get('X-API-Key') or ''
        a = self.headers.get('Authorization') or ''
        if h == API_KEY or a == 'Bearer ' + API_KEY:
            return True
        self._send(401, {'error': 'unauthorized：请带 X-API-Key 或 Authorization: Bearer <key>'})
        return False

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode('utf-8'))
        except Exception:
            return {}

    # ---------- GET ----------
    def do_OPTIONS(self):
        """[2026-10-04] CORS 预检：浏览器跨域 POST(JSON) 前会先发 OPTIONS，之前返回 501 ⇒ 前端 Failed to fetch"""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        path = p.path
        if path in ('/health', '/v1/health'):
            return self._send(200, {
                'ok': True, 'uptime_s': round(time.time() - S['started'], 1),
                'klein_loaded': S['klein'] is not None,
                'klein_mode': S.get('klein_mode'),
                'jobs': {k: v['status'] for k, v in S['jobs'].items()},
                'output_dir': OUT_DIR, 'auth': bool(API_KEY)})
        if path == '/v1/jobs':
            return self._send(200, {'jobs': [{k: v.get(k) for k in
                                              ('job_id', 'status', 'created', 'finished', 'error', 'params')}
                                             | ({'result': v['result']} if v.get('result') else {})
                                             for v in S['jobs'].values()]})
        if path.startswith('/v1/video'):
            jid = path.rsplit('/', 1)[-1]
            if jid in ('video', ''):
                jid = urllib.parse.parse_qs(p.query).get('id', [''])[0]
            job = S['jobs'].get(jid)
            if not job:
                return self._send(404, {'error': 'no such job', 'job_id': jid})
            out = {k: job.get(k) for k in ('job_id', 'status', 'error', 'params')}
            out['result'] = job.get('result')          # [2026-10-04] 前端需要拿结果文件
            out['video'] = job.get('video') or (job.get('result') or {}).get('video')
            out['file'] = job.get('file') or (job.get('result') or {}).get('file')
            if job.get('result'):
                r = dict(job['result'])
                r['url'] = self._file_url(r['filename'])
                out['result'] = r
            return self._send(200, out)
        if path.startswith('/files/'):
            name = os.path.basename(urllib.parse.unquote(path[len('/files/'):]))
            fp = os.path.join(OUT_DIR, name)
            if not os.path.exists(fp):
                return self._send(404, {'error': 'not found', 'file': name})
            ctype = 'video/mp4' if name.endswith('.mp4') else (
                'image/png' if name.endswith('.png') else 'application/octet-stream')
            with open(fp, 'rb') as f:
                data = f.read()
            return self._send(200, data, ctype)
        if path == '/':
            return self._send(200, {'name': 'gen_api', 'endpoints': [
                'GET /health', 'POST /v1/image', 'POST /v1/images/generations',
                'POST /v1/video', 'GET /v1/video/<job_id>', 'GET /v1/jobs',
                'POST /v1/free', 'GET /files/<filename>']})
        return self._send(404, {'error': 'not found', 'path': path})

    # ---------- POST ----------
    def do_POST(self):
        if not self._auth_ok():
            return
        p = urllib.parse.urlparse(self.path)
        path = p.path
        body = self._body()
        try:
            if path == '/v1/image':
                prompt = (body.get('prompt') or '').strip()
                if not prompt:
                    return self._send(400, {'error': 'missing prompt'})
                r = gen_image(prompt, int(body.get('width', 864)), int(body.get('height', 1536)),   # 不给尺寸就用默认竖版 864x1536
                              int(body.get('steps', 8)),           # [2026-10-15 主人令] 蒸馏版 8 步（原 28 步纯浪费）
                              body.get('seed'), body.get('filename'))
                _sr = int(body.get('sr', 0) or 0)                      # [2026-10-15 主人令] 默认不超分（要超分再喊 sr=2/4）
                if _sr > 0:
                    try:
                        import upscale_lib
                        u = upscale_lib.upscale_image(r['file'], scale=_sr)
                        r['sr'] = _sr
                        r['sr_file'] = u['file']
                        r['file'] = u['file']
                        r['filename'] = os.path.basename(u['file'])
                        r['width'], r['height'] = u['width'], u['height']
                        r['sr_elapsed'] = u['elapsed']
                    except Exception as e:
                        r['sr_error'] = str(e)[:200]
                r['url'] = self._file_url(r['filename'])
                _trim_memory('image')     # [2026-10-01] 生图收尾回收
                _mem_guard('image')
                return self._send(200, r)
            if path == '/v1/upscale':
                f = (body.get('file') or body.get('path') or '').strip()
                if not f or not os.path.exists(f):
                    return self._send(400, {'error': 'file 不存在: %s' % f})
                r = gen_upscale(f, int(body.get('scale', 4)))
                r['url'] = self._file_url(os.path.basename(r['file']))
                _trim_memory('upscale')   # [2026-10-01]
                _mem_guard('upscale')
                return self._send(200, r)
            if path == '/v1/images/generations':
                prompt = (body.get('prompt') or '').strip()
                if not prompt:
                    return self._send(400, {'error': 'missing prompt'})
                w, h = 864, 1536                                   # 默认竖版；横版传 size=1536x864
                size = str(body.get('size') or '864x1536')
                if 'x' in size:
                    try:
                        w, h = [int(x) for x in size.lower().split('x')[:2]]
                    except Exception:
                        pass
                r = gen_image(prompt, w, h, int(body.get('steps', 20)), body.get('seed'))
                item = {'url': self._file_url(r['filename']), 'revised_prompt': prompt,
                        'local_path': r['file']}
                if body.get('response_format') == 'b64_json':
                    import base64
                    with open(r['file'], 'rb') as f:
                        item['b64_json'] = base64.b64encode(f.read()).decode()
                return self._send(200, {'created': int(time.time()), 'data': [item]})
            if path == '/v1/video':
                prompt = (body.get('prompt') or '').strip()
                if not prompt:
                    return self._send(400, {'error': 'missing prompt'})
                params = {'prompt': prompt,
                          'width': int(body.get('width', 768)),
                          'height': int(body.get('height', 512)),
                          'frames': int(body.get('frames', 97)),
                          'upscale': bool(body.get('upscale', False)),
                          'seed': int(body.get('seed', 42))}
                if int(body.get('sr', 4) or 0) > 0:              # 默认 4 倍超分
                    params['sr'] = int(body.get('sr', 4))
                _img = (body.get('image_path') or '').strip()
                if _img:
                    if not os.path.exists(_img):
                        return self._send(400, {'error': 'image_path 不存在: %s' % _img})
                    params['image_path'] = _img
                jid = new_video_job(params)
                return self._send(200, {'job_id': jid, 'status': 'queued',
                                        'poll': 'GET /v1/video/%s' % jid, 'params': params})
            if path.startswith('/v1/cancel'):
                jid = path.rsplit('/', 1)[-1]
                job = S['jobs'].get(jid)
                if not job:
                    return self._send(404, {'error': 'no such job', 'job_id': jid})
                job['cancelled'] = True
                if job.get('status') == 'queued':
                    job['status'] = 'cancelled'
                return self._send(200, {'job_id': jid, 'cancelled': True, 'status': job.get('status')})
            if path == '/v1/free':
                freed = unload_klein('手动 /v1/free')
                comfy = False
                try:
                    import ltx_client
                    comfy = ltx_client.free_comfy()
                except Exception:
                    pass
                return self._send(200, {'klein_unloaded': freed, 'comfy_freed': comfy})
        except Exception as e:
            log('处理 %s 异常: %s\n%s' % (path, str(e)[:200], traceback.format_exc()[-800:]))
            return self._send(500, {'error': str(e)[:500], 'path': path})
        return self._send(404, {'error': 'not found', 'path': path})

    def _file_url(self, name):
        host = self.headers.get('Host') or ('127.0.0.1:%d' % PORT)
        return 'http://%s/files/%s' % (host, urllib.parse.quote(name))


def selftest():
    import urllib.request
    base = 'http://127.0.0.1:%d' % PORT

    def post(path, payload):
        req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(req, timeout=900).read())

    def get(path):
        return json.loads(urllib.request.urlopen(base + path, timeout=900).read())

    print('health:', get('/health'))
    print('image:', post('/v1/image', {'prompt': 'a red panda on a mossy rock, morning light',
                                       'width': 768, 'height': 768, 'steps': 20}))
    print('health2:', get('/health'))


def main():
    if '--selftest' in sys.argv:
        selftest(); return
    threading.Thread(target=idle_watchdog, daemon=True).start()
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    log('gen_api 已启动：http://127.0.0.1:%d  (输出目录 %s, 鉴权=%s)' % (PORT, OUT_DIR, bool(API_KEY)))
    srv.serve_forever()


if __name__ == '__main__':
    main()
