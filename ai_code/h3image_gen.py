"""MiniMax-H3-Image 生成脚本（diffusers Modular 路线，只写未跑）。

口径与官方一致：固定 6 步（PDD 头），无 guidance_scale / negative_prompt / steps 参数。
需要 2026-11-10 之后的 diffusers main（见 D:\\ai\\H3IMAGE_工作流.md 第三节）。
本机 5070 12GB 放不下整组件 DiT（66GB），此路线面向大显存卡/云机；
本地路线等三层流式移植（工作流第二节路线 C）。

用法:
  python h3image_gen.py --prompt "..." --size 16:9 --seed 42
  python h3image_gen.py --prompt "Keep everything in <Picture 1> ..." --ref a.png --size 1:1
  python h3image_gen.py --prompt "..." --dry_run     # 只打印计划，不加载模型
"""
import argparse
import os
import sys
import time

MODEL_DEFAULT = r'D:\ai\models\MiniMax-H3-Image-diffusers'
OUT_DIR = r'D:\ai\output\images'
SIZES = {
    '1:1': (4096, 4096),
    '4:3': (4736, 3552),
    '3:4': (3552, 4736),
    '3:2': (5024, 3360),
    '2:3': (3360, 5024),
    '16:9': (5472, 3072),
    '9:16': (3072, 5472),
    '21:9': (6272, 2688),
}


def parse_size(s):
    if s in SIZES:
        return SIZES[s]
    w, h = s.lower().split('x')
    w, h = int(w), int(h)
    if w % 32 or h % 32:
        sys.exit(f'RESULT: FAIL 宽高必须是 32 的倍数: {w}x{h}')
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', help='提示词文本（详细英文描述，官方口径）')
    ap.add_argument('--prompt_file', help='从文件读提示词（与 --prompt 二选一）')
    ap.add_argument('--ref', action='append', default=[],
                    help='参考图路径，可多次；顺序 = 提示词里的 <Picture 1>...，最多 16 张')
    ap.add_argument('--size', default='1:1', help='预设比例或 WxH（32 的倍数，4-20MP）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', default=MODEL_DEFAULT, help='转换后的 diffusers 布局目录')
    ap.add_argument('--offload', action='store_true',
                    help='整组件 CPU offload（DiT 是 66GB 单组件，12GB 卡无效）')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry_run', action='store_true', help='只打印计划，不加载模型')
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding='utf-8') as f:
            prompt = f.read().strip()
    if not prompt:
        sys.exit('RESULT: FAIL 需要 --prompt 或 --prompt_file')
    w, h = parse_size(args.size)
    mp = w * h / 1e6
    if len(args.ref) > 16:
        sys.exit('RESULT: FAIL 参考图最多 16 张')
    for r in args.ref:
        if not os.path.isfile(r):
            sys.exit(f'RESULT: FAIL 参考图不存在: {r}')

    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(
        OUT_DIR, f'h3image_{time.strftime("%Y%m%d_%H%M%S")}_{args.seed}.png')

    print(f'[plan] model={args.model}', flush=True)
    print(f'[plan] size={w}x{h} ({mp:.1f}MP) seed={args.seed} refs={len(args.ref)} offload={args.offload}', flush=True)
    print(f'[plan] steps=6(固定) guidance=无(已烘进权重) out={out}', flush=True)
    print(f'[plan] prompt({len(prompt)}字符): {prompt[:120]}...', flush=True)
    if mp < 4 or mp > 20:
        print(f'[warn] {mp:.1f}MP 超出官方有效区间 4-20MP', flush=True)
    if args.dry_run:
        print('RESULT: DRY_RUN 未加载模型', flush=True)
        return

    try:
        import torch
        from diffusers import ModularPipeline
    except ImportError as e:
        sys.exit(f'RESULT: FAIL 导入失败({e})；需要 diffusers main(>=2026-11-10)，'
                 f'建 venv 步骤见 D:\\ai\\H3IMAGE_工作流.md 第三节')
    if not hasattr(ModularPipeline, 'from_pretrained'):
        sys.exit('RESULT: FAIL diffusers 版本过旧，无 ModularPipeline')

    t0 = time.time()
    manager = None
    if args.offload:
        from diffusers import ComponentsManager
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device='cuda')
    pipe = ModularPipeline.from_pretrained(args.model, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)
    print(f'[load] {time.time()-t0:.1f}s', flush=True)

    kw = dict(prompt=prompt, height=h, width=w,
              generator=torch.Generator(device='cpu').manual_seed(args.seed),
              output='images')
    if args.ref:
        from diffusers.utils import load_image
        kw['image'] = [load_image(r) for r in args.ref]

    t1 = time.time()
    result = pipe(**kw)
    img = result[0]
    img.save(out)
    print(f'RESULT: OK file={out} size={img.size} seed={args.seed} '
          f'gen={time.time()-t1:.1f}s total={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()