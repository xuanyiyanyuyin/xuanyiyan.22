# -*- coding: utf-8 -*-
"""LTX-2.5 diffusion-VAE decode SPEED TEST (no video output)
python speed_test_decode.py [F] [Hlat] [Wlat]   default: 9 32 32 -> 1024x1024 video, 1 tile

2K (2560x1408) 解码 = 6 块 1024x1024 tile, 每块规格与本测试的 1 块相同
-> 2K 总时长 ~= 单块耗时 x 6
"""
import sys, os, time, torch

sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE'] = '1'
VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'


def main(F, H, W):
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, video_decoder_sd_ops_for_checkpoint
    from ltx_core.model.video_vae.diffusion_tiling import DiffVAEMode
    from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op
    from ltx_core.tiling import TileSizeConfig, DimensionSizeConfig

    t0 = time.time()
    builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=VAE_FILE,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE_FILE),
        module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER),),
    )
    decoder = builder.build(dtype=torch.bfloat16)
    print(f'[load] VAE 417M loaded {time.time() - t0:.1f}s', flush=True)

    latent = torch.randn(1, 128, F, H, W, dtype=torch.bfloat16, device='cuda')
    print(f'[in] latent={tuple(latent.shape)}  video={W * 32}x{H * 32}  ~{(4 * F - 3) * 2} frames', flush=True)

    tc = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=0, overlap=0),
        height=DimensionSizeConfig(tile_size=1024, overlap=160),
        width=DimensionSizeConfig(tile_size=1024, overlap=160),
    )

    # ---- progress hooks ----
    import ltx_core.model.video_vae.diffusion_tiling as _dt
    tot = {'n': 0}
    _op = _dt.prepare_tile_schedule

    def _prep(*a, **kw):
        r = _op(*a, **kw)
        tot['n'] = len(r)
        print(f'[tiling] {len(r)} tiles (1024x1024 grid, overlap 160)', flush=True)
        return r

    _dt.prepare_tile_schedule = _prep

    prog = {'i': 0, 't': time.time()}
    _ot = decoder._decode_one_tile

    def _tile(*a, **kw):
        prog['i'] += 1
        k, n = prog['i'], max(tot['n'], 1)
        ts = time.time()
        extra = f', last {(ts - prog["t"]) / (k - 1):.1f}s' if k > 1 else ''
        print(f'  [tile {k}/{n}] start  elapsed {ts - prog["t"]:.1f}s{extra}', flush=True)
        r = _ot(*a, **kw)
        print(f'  [tile {k}/{n}] done   {time.time() - ts:.1f}s', flush=True)
        return r

    decoder._decode_one_tile = _tile

    t1 = time.time()
    with torch.inference_mode():
        chunks = 0
        for ch in decoder.decode_video(latent, tiling_config=tc):
            chunks += 1
            print(f'[chunk {chunks}] out {tuple(ch.shape)}', flush=True)
    dt = time.time() - t1
    n = max(tot['n'], 1)
    print(f'[RESULT] tiles={n} chunks={chunks} total={dt:.1f}s  per_tile={dt / n:.1f}s', flush=True)
    print(f'[EXTRAPOLATE] 2K 2560x1408 (6 tiles) ~= {dt / n * 6:.0f}s = {dt / n * 6 / 60:.1f} min', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(int(a[0]) if len(a) > 0 else 9,
         int(a[1]) if len(a) > 1 else 32,
         int(a[2]) if len(a) > 2 else 32)
