# -*- coding: utf-8 -*-
"""LTX-2.5 DiffVAE 手动开 torch.compile 测速
基准: 同形状单块 eager = 282.2s (speed_test_decode.py)
本脚本只改一件事: 补上被 resolve_attention_for_host 强制跳过的 compile_diffusion_decoder
run1 含编译开销, run2 为稳态; 两次输出做数值对比
"""
import sys, os, time, torch

sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE'] = '1'
VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'
BASE = 282.2  # eager 基准 (s/tile)


def main():
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, video_decoder_sd_ops_for_checkpoint
    from ltx_core.model.video_vae.diffusion_tiling import DiffVAEMode
    from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op
    from ltx_core.model.video_vae.transformer.compiling import compile_diffusion_decoder
    from ltx_core.model.video_vae.transformer.chunked.block import ChunkedDiffusionNABlock
    from ltx_core.tiling import TileSizeConfig, DimensionSizeConfig

    t0 = time.time()
    decoder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=VAE_FILE,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE_FILE),
        module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER),),
    ).build(dtype=torch.bfloat16)
    print(f'[load] {time.time() - t0:.1f}s', flush=True)

    # ★ 唯一改动: 手动补上 compile (基准脚本里这步被强制跳过)
    t = time.time()
    compile_diffusion_decoder(decoder, compile_det_stages=False)
    n = sum(1 for b in decoder.diff_blocks if isinstance(b, ChunkedDiffusionNABlock))
    print(f'[compile] wrapped {n} blocks .forward_attn_mlp, mark_dynamic_shapes={decoder.mark_dynamic_shapes} '
          f'({time.time() - t:.2f}s)', flush=True)

    latent = torch.randn(1, 128, 9, 32, 32, dtype=torch.bfloat16, device='cuda')
    print(f'[in] latent={tuple(latent.shape)} video=1024x1024', flush=True)
    tc = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=0, overlap=0),
        height=DimensionSizeConfig(tile_size=1024, overlap=160),
        width=DimensionSizeConfig(tile_size=1024, overlap=160),
    )

    import ltx_core.model.video_vae.diffusion_tiling as _dt
    tot = {'n': 0}
    _op = _dt.prepare_tile_schedule

    def _prep(*a, **kw):
        r = _op(*a, **kw)
        tot['n'] = len(r)
        print(f'[tiling] {len(r)} tiles', flush=True)
        return r

    _dt.prepare_tile_schedule = _prep

    prog = {'i': 0}

    def _install_hook():
        prog['i'] = 0
        _ot = decoder._decode_one_tile

        def _tile(*a, **kw):
            prog['i'] += 1
            ts = time.time()
            r = _ot(*a, **kw)
            print(f'    [tile {prog["i"]}/{max(tot["n"], 1)}] {time.time() - ts:.1f}s', flush=True)
            return r

        decoder._decode_one_tile = _tile

    outs = []
    times = []
    for run in (1, 2):
        _install_hook()
        t = time.time()
        try:
            with torch.inference_mode():
                chunks = [c for c in decoder.decode_video(latent, tiling_config=tc)]
        except torch.OutOfMemoryError as e:
            print(f'[run{run}] OOM: {str(e)[:200]}', flush=True)
            return
        dt = time.time() - t
        times.append(dt)
        o = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        outs.append(o.float().cpu())
        print(f'[run{run}] total {dt:.1f}s  out {tuple(o.shape)}  peakVRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
        del chunks, o
        torch.cuda.empty_cache()

    d = (outs[0] - outs[1]).abs()
    print(f'[diff run1 vs run2] max={d.max():.6f} mean={d.mean():.8f}', flush=True)
    print(f'[RESULT] run1(with compile)={times[0]:.1f}s  run2(steady)={times[1]:.1f}s  '
          f'eager_baseline={BASE}s  speedup={BASE / times[1]:.2f}x', flush=True)


if __name__ == '__main__':
    main()
