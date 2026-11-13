# -*- coding: utf-8 -*-
"""4x-UltraSharp 超分 (spandrel 加载, 分块推理, 支持 RGBA 透明通道)
用法: python upscale4x.py <输入图> [输出图] [--tile 768]
默认输出 = 输入名_4x.png
"""
import sys, os, time
import numpy as np
import torch
from PIL import Image
import spandrel

MODEL = r'D:\ai\models\upscale\4x-UltraSharp.safetensors'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def load():
    md = spandrel.ModelLoader().load_from_file(MODEL)
    m = md.model.eval().to(DEV)
    scale = getattr(md, 'scale', 4) or 4
    return m, int(scale)

@torch.no_grad()
def run(model, scale, arr, tile=768, pad=24):
    """arr: HxWxC float32 [0,1] → 4x 后的同通道数数组"""
    h, w, c = arr.shape
    out = np.zeros((h * scale, w * scale, c), np.float32)
    ys = list(range(0, h, tile)); xs = list(range(0, w, tile))
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            y0, y1 = max(0, y - pad), min(h, y + tile + pad)
            x0, x1 = max(0, x - pad), min(w, x + tile + pad)
            t = torch.from_numpy(arr[y0:y1, x0:x1]).permute(2, 0, 1).unsqueeze(0).to(DEV)
            if t.shape[1] == 4:                      # RGBA → 前3通道过模型, A 用双三次
                rgb = model(t[:, :3])[0].clamp(0, 1)
                a = torch.nn.functional.interpolate(t[:, 3:4], scale_factor=scale, mode='bicubic', align_corners=False)[0]
                o = torch.cat([rgb, a], 0)
            else:
                o = model(t)[0].clamp(0, 1)
            o = o.permute(1, 2, 0).cpu().numpy()
            oy0, ox0 = (y - y0) * scale, (x - x0) * scale
            oh, ow = min(tile, h - y) * scale, min(tile, w - x) * scale
            out[y*scale:y*scale+oh, x*scale:x*scale+ow] = o[oy0:oy0+oh, ox0:ox0+ow]
            print(f'  tile {yi*len(xs)+xi+1}/{len(ys)*len(xs)} ({x},{y})', flush=True)
    return out

def main():
    if len(sys.argv) < 2:
        print('用法: python upscale4x.py <输入图> [输出图]'); return
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else \
          os.path.splitext(src)[0] + '_4x.png'
    tile = 768
    if '--tile' in sys.argv:
        tile = int(sys.argv[sys.argv.index('--tile') + 1])
    t0 = time.time()
    im = Image.open(src)
    arr = np.array(im.convert('RGBA' if im.mode == 'RGBA' else 'RGB')).astype(np.float32) / 255.0
    print(f'输入 {im.size} mode={im.mode} → 加载 4x-UltraSharp', flush=True)
    model, scale = load()
    print(f'模型 scale={scale}, 开始分块({tile}px)推理...', flush=True)
    out = run(model, scale, arr, tile=tile)
    res = Image.fromarray((out * 255).round().clip(0, 255).astype(np.uint8), 'RGBA' if arr.shape[2] == 4 else 'RGB')
    res.save(dst)
    print(f'完成 {res.size}  用时 {time.time()-t0:.1f}s → {dst}', flush=True)

if __name__ == '__main__':
    main()
