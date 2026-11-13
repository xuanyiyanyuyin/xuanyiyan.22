# -*- coding: utf-8 -*-
"""验证 torch.compile 版 DiffVAE
1) 数值对齐: 同一小 latent, eager vs compile 输出对比
2) 多块连续: 2 块 tile 连续 decode, 看第 2 块会不会像上次那样卡死 + 测速
"""
import sys, os, time, torch

sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE'] = '1'
VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'


def build_decoder():
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, video_decoder_sd_ops_for_checkpoint
    from ltx_core.model.video_vae.diffusion_tiling import DiffVAEMode
    from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op
    return SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=VAE_FILE,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE_FILE),
        module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER),),
    ).build(dtype=torch.bfloat16)


def make_tc():
    from ltx_core.tiling import TileSizeConfig, DimensionSizeConfig
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=0, overlap=0),
        height=DimensionSizeConfig(tile_size=1024, overlap=160),
        width=DimensionSizeConfig(tile_size=1024, overlap=160),
    )


def install_hook(decoder):
    import ltx_core.model.video_vae.diffusion_tiling as _dt
    tot = {'n': 0}
    if not getattr(_dt, '_hooked', False):
        _op = _dt.prepare_tile_schedule

        def _prep(*a, **kw):
            r = _op(*a, **kw)
            tot['n'] = len(r)
            print(f'  [tiling] {len(r)} tiles', flush=True)
            return r

        _dt.prepare_tile_schedule = _prep
        _dt._hooked = True
    prog = {'i': 0}
    _ot = decoder._decode_one_tile

    def _tile(*a, **kw):
        prog['i'] += 1
        ts = time.time()
        r = _ot(*a, **kw)
        print(f'  [tile {prog["i"]}/{max(tot["n"], 1)}] {time.time() - ts:.1f}s', flush=True)
        return r

    decoder._decode_one_tile = _tile
    return tot


def run(decoder, latent, tag, tc=None):
    t = time.time()
    with torch.inference_mode():
        chunks = [c for c in decoder.decode_video(latent, tiling_config=tc)]
    o = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    print(f'[{tag}] {time.time() - t:.1f}s out={tuple(o.shape)} peakVRAM={torch.cuda.max_memory_allocated() / 1e9:.2f}GB', flush=True)
    return o


def main():
    from ltx_core.model.video_vae.transformer.compiling import compile_diffusion_decoder

    print('== build (eager) ==', flush=True)
    decoder = build_decoder()
    install_hook(decoder)

    # ---------- 1) 数值对齐: 小 latent ----------
    small = torch.randn(1, 128, 5, 8, 8, dtype=torch.bfloat16, device='cuda')  # 256x256, 17帧
    print(f'== 1) eager vs compile 对齐 (latent {tuple(small.shape)}) ==', flush=True)
    o_eager = run(decoder, small, 'eager-small')

    t = time.time()
    compile_diffusion_decoder(decoder, compile_det_stages=False)
    print(f'[compile] wrapped (compile_det_stages=False) {time.time() - t:.2f}s', flush=True)

    o_comp = run(decoder, small, 'compile-small')
    d = (o_eager.float() - o_comp.float()).abs()
    verdict = 'OK-aligned' if d.max() < 5e-3 else 'WARN-deviation'
    print(f'[ALIGN] max={d.max():.6f} mean={d.mean():.8f}  {verdict}', flush=True)
    del o_eager, o_comp, d
    torch.cuda.empty_cache()

    # ---------- 2) 单块测速 (1024x1024, 与 eager 基准 282.2s 同规格) ----------
    print('== 2) 单块 decode 测速 (1024x1024 tile) ==', flush=True)
    big = torch.randn(1, 128, 9, 32, 32, dtype=torch.bfloat16, device='cuda')  # 1024x1024 -> 1 tile
    t = time.time()
    o_big = run(decoder, big, 'compile-1tile', tc=make_tc())
    dt = time.time() - t
    print(f'[SPEEDUP] eager 282.2s -> compile {dt:.1f}s  ({282.2 / dt:.2f}x)', flush=True)
    print('[DONE]', flush=True)


if __name__ == '__main__':
    main()
