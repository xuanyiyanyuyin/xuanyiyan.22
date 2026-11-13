"""LTX-2.5 官方管线 hook — 抓取 transformer 真实输入 + 每块输出
用法: python ltx25_capture_official.py [--prompt ...] [--steps 4]
产出: D:\ai\official_trace\inputs.pt + block{NN}_out.pt
注意: 需按实际加载方式调整 (diffusers / ltx-pipelines)
"""
import torch, argparse, os
from pathlib import Path

SAVE_DIR = Path(r"D:\ai\official_trace")
SAVE_DIR.mkdir(exist_ok=True)


def install_hooks(transformer, save_dir):
    """在 transformer 入口 + 每块输出挂 hook"""
    # 1. 入口 hook (diffusers 风格: hidden_states/timestep/encoder_hidden_states)
    def input_hook(module, args, kwargs):
        data = {
            'hidden_states': kwargs.get('hidden_states', args[0] if len(args) > 0 else None),
            'timestep': kwargs.get('timestep', args[1] if len(args) > 1 else None),
            'sigma': kwargs.get('sigma', None),
            'encoder_hidden_states': kwargs.get('encoder_hidden_states', args[2] if len(args) > 2 else None),
            'num_frames': kwargs.get('num_frames', None),
            'height': kwargs.get('height', None),
            'width': kwargs.get('width', None),
        }
        to_save = {k: v.detach().cpu() for k, v in data.items() if v is not None}
        torch.save(to_save, save_dir / "inputs.pt")
        print(f"[HOOK] inputs saved: {list(to_save.keys())}")
        for k, v in to_save.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} dtype={v.dtype} std={v.float().std().item():.4f}")

    try:
        transformer.register_forward_pre_hook(input_hook, with_kwargs=True)
        print("[HOOK] transformer 入口 hook 安装 OK")
    except Exception as e:
        print(f"[HOOK] 入口 hook 失败: {e}")

    # 2. 每块输出 hook
    blocks = None
    for attr in ['transformer_blocks', 'transformer_blocks_']:
        if hasattr(transformer, attr):
            blocks = getattr(transformer, attr)
            break
    if blocks is None and hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    if blocks is None:
        print("[HOOK] 找不到 transformer_blocks, 请检查 transformer 结构")
        return

    for i, block in enumerate(blocks):
        def make_hook(idx):
            def hook(module, args, kwargs, output):
                out = output if isinstance(output, torch.Tensor) else output[0]
                torch.save(out.detach().cpu(), save_dir / f"block{idx:02d}_out.pt")
                s = out.float().std().item(); m = out.float().abs().max().item()
                print(f"[HOOK] block{idx:02d}: std={s:.4f} max={m:.4f}")
            return hook
        block.register_forward_hook(make_hook(i), with_kwargs=True)
    print(f"[HOOK] {len(blocks)} 块 hook 安装 OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='a cat walking in a garden')
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--model_path', default=r'D:\ai\models\LTX-2.5')
    args = ap.parse_args()

    # 加载方式 A: diffusers (若 LTX-2.5 被 diffusers 支持)
    try:
        from diffusers import LTXPipeline
        print("[load] 尝试 diffusers LTXPipeline...")
        pipe = LTXPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        pipe.to('cuda')
        install_hooks(pipe.transformer, SAVE_DIR)
        print("[load] diffusers 加载成功")
    except Exception as e:
        print(f"[load] diffusers 失败: {e}")
        print("[load] 尝试方式 B: ltx-pipelines (需手动调整)")
        # 方式 B: ltx-pipelines (LTX-2 repo)
        # from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        # pipeline = TI2VidOneStagePipeline(model_paths=..., ...)
        # 在 pipeline 内部 transformer 上挂 hook
        raise SystemExit("[load] 请调整加载方式 (diffusers 或 ltx-pipelines)")

    # 运行
    gen = torch.Generator('cuda').manual_seed(42)
    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        width=256, height=256, num_frames=9,
        generator=gen,
    )
    print(f"\n[trace] saved to {SAVE_DIR}")
    print(f"  files: {sorted(os.listdir(SAVE_DIR))}")


if __name__ == '__main__':
    main()
