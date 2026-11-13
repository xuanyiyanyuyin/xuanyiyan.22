# -*- coding: utf-8 -*-
"""超分（图片 / 视频）—— 基于本地 4x-UltraSharp（spandrel 加载）+ ffmpeg 重新封装

对外：
    upscale_image(src, dst=None, scale=4, tile=768)      -> 输出路径
    upscale_video(src, dst=None, scale=2, tile=512, crf=16) -> 输出路径（保留音轨）
    ffmpeg_path()                                        -> ffmpeg.exe 路径
    model_scale()                                        -> 模型原生倍率（4）
说明：模型原生 4x；要 2x 就先 4x 再高质量降采样（比直接 2x 模型干净）。
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image

MODELS = {
    'image': r'D:\ai\models\upscale\4x-UltraSharp.safetensors',   # 图片主力：4x，细节强
    'video': r'D:\ai\models\upscale\RealESRGAN_x2.pth',           # 视频主力：Real-ESRGAN x2（原生 2 倍，速度快、时序更稳）
    'x4': r'D:\ai\models\upscale\RealESRGAN_x4.pth',              # 备选：Real-ESRGAN x4
}
MODEL = MODELS['image']       # 兼容旧调用
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FFMPEG_CANDIDATES = [
    r'C:\Users\Jinqi Wei\AppData\Local\Microsoft\WindowsApps\ffmpeg.exe',
    r'D:\ai\napcat_new\napcat\ffmpeg.exe',
    r'D:\ai\python\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe',
]

_CACHE = {}


def ffmpeg_path():
    for p in FFMPEG_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load(which='image'):
    """按用途取模型（image/video/x4），带缓存"""
    path = MODELS.get(which, MODELS['image'])
    if path not in _CACHE:
        import spandrel
        md = spandrel.ModelLoader().load_from_file(path)
        _CACHE[path] = (md.model.eval().to(DEV), int(getattr(md, 'scale', 4) or 4), path)
    return _CACHE[path]


def model_scale(which='image'):
    return _load(which)[1]


def _run_tiled(model, scale, arr, tile=768, pad=24):
    """arr: HxWxC float32[0,1] → scale 倍后的数组（分块推理，省显存）"""
    h, w, c = arr.shape
    out = np.zeros((h * scale, w * scale, c), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, y1 = max(0, y - pad), min(h, y + tile + pad)
            x0, x1 = max(0, x - pad), min(w, x + tile + pad)
            t = torch.from_numpy(arr[y0:y1, x0:x1]).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                if t.shape[1] == 4:
                    rgb = model(t[:, :3])[0].clamp(0, 1)
                    a = torch.nn.functional.interpolate(t[:, 3:4], scale_factor=scale,
                                                        mode='bicubic', align_corners=False)[0]
                    o = torch.cat([rgb, a], 0)
                else:
                    o = model(t)[0].clamp(0, 1)
            o = o.permute(1, 2, 0).cpu().numpy()
            oy0, ox0 = (y - y0) * scale, (x - x0) * scale
            oh = min(tile, h - y) * scale
            ow = min(tile, w - x) * scale
            out[y * scale:y * scale + oh, x * scale:x * scale + ow] = o[oy0:oy0 + oh, ox0:ox0 + ow]
    return out


def _arr_of(img):
    a = np.asarray(img).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def _to_pil(arr):
    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def upscale_image(src, dst=None, scale=4, tile=768, which='image'):
    """图片超分；scale 可为 2/4（4x-UltraSharp 原生 4x，要 2x 就 4x 后降采样）"""
    model, ns, mpath = _load(which)
    img = Image.open(src)
    t0 = time.time()
    arr = _arr_of(img)
    big = _run_tiled(model, ns, arr, tile=tile)
    out = _to_pil(big)
    if scale and scale != ns:
        w = int(img.width * scale)
        h = int(img.height * scale)
        out = out.resize((w, h), Image.LANCZOS)
    if not dst:
        root, ext = os.path.splitext(src)
        dst = '%s_%dx%s' % (root, scale or ns, ext or '.png')
    out.save(dst)
    return {'file': dst, 'src': src, 'scale': scale or ns, 'width': out.width, 'height': out.height,
            'elapsed': round(time.time() - t0, 1)}


def upscale_video(src, dst=None, scale=4, tile=384, crf=16, on_progress=None, which=None):
    """视频超分：逐帧过模型 → ffmpeg 重新编码（保留原音轨）
    scale>=4 用 Real-ESRGAN x4，scale=2 用 x2（快一倍）"""
    import av
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError('找不到 ffmpeg.exe')
    if which is None:
        which = 'x4' if (scale or 2) >= 4 else 'video'
    model, ns, mpath = _load(which)
    tmp = tempfile.mkdtemp(prefix='vsr_')
    t0 = time.time()
    try:
        cin = av.open(src)
        vstream = next(s for s in cin.streams if s.type == 'video')
        fps = float(vstream.average_rate) if vstream.average_rate else 24.0
        n = 0
        for frame in cin.decode(video=0):
            img = frame.to_image()
            big = _to_pil(_run_tiled(model, ns, _arr_of(img), tile=tile))
            if scale and scale != ns:
                big = big.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            big.save(os.path.join(tmp, '%05d.png' % n))
            n += 1
            if on_progress and n % 10 == 0:
                on_progress(n)
        cin.close()
        if not dst:
            root, _ = os.path.splitext(src)
            dst = '%s_%dx.mp4' % (root, scale or ns)
        cmd = [ff, '-y', '-framerate', '%.6f' % fps, '-i', os.path.join(tmp, '%05d.png'),
               '-i', src, '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-crf', str(crf),
               '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-shortest', dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError('ffmpeg 编码失败: %s' % (r.stderr or '')[-300:])
        probe = av.open(dst)
        st = next(s for s in probe.streams if s.type == 'video')
        info = {'width': st.width, 'height': st.height}
        probe.close()
        return {'file': dst, 'src': src, 'scale': scale or ns, 'frames': n,
                'width': info['width'], 'height': info['height'],
                'elapsed': round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    import json
    if len(sys.argv) < 3:
        print('用法: python upscale_lib.py image|video <输入> [scale]')
        sys.exit(0)
    kind, path = sys.argv[1], sys.argv[2]
    sc = int(sys.argv[3]) if len(sys.argv) > 3 else (4 if kind == 'image' else 2)
    fn = upscale_image if kind == 'image' else upscale_video
    print(json.dumps(fn(path, scale=sc), ensure_ascii=False))
