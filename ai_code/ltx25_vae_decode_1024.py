"""LTX-2.5 官方 VAE decode — latent(1,128,F,H,W) → 视频帧 → mp4
用法: python ltx25_vae_decode_1024.py [latent.pt]
"""
import sys, os, time, torch
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
os.environ['HF_HUB_OFFLINE'] = '1'

VAE_FILE = r'D:\ai\_ltx25_src\ltx-2.5-video-vae-bf16.safetensors'

def main(latent_path):
    from ltx_core.loader import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, video_decoder_sd_ops_for_checkpoint
    from ltx_core.model.video_vae.diffusion_tiling import DiffVAEMode
    from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op

    latent = torch.load(latent_path, weights_only=True)  # (1,128,F,H,W)
    if isinstance(latent, dict):
        latent = latent.get('latent', latent.get('latents'))
    print(f'latent: {tuple(latent.shape)}', flush=True)
    F = latent.shape[2]

    t0 = time.time()
    builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=VAE_FILE,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE_FILE),
        module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER),),
    )
    decoder = builder.build(dtype=torch.bfloat16)
    n_param = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f'VAE 加载 {n_param:.0f}M 参数 {time.time()-t0:.0f}s', flush=True)

    # ★ 补上官方因缺 natten 被 resolve_attention_for_host() 连带跳过的 torch.compile
    #   (即官方 chunked_compile 预设; 实测单块 282s -> 75s, 2026-09-11)
    if os.environ.get('LTX_VAE_COMPILE', '1') == '1':
        from ltx_core.model.video_vae.transformer.compiling import compile_diffusion_decoder
        compile_diffusion_decoder(decoder, compile_det_stages=False)
        print('[compile] torch.compile ON (chunked, compile_det_stages=False)', flush=True)

    latent = latent.to('cuda').bfloat16()
    t1 = time.time()

    # ---- 进度钩子 (2026-09-11): 打印总分块数 + 每块开始/完成 + ETA ----
    import ltx_core.model.video_vae.diffusion_tiling as _dt
    _tot = {'n': 0, 'grp': 0}
    _orig_prep = _dt.prepare_tile_schedule

    def _prep(*a, **kw):
        r = _orig_prep(*a, **kw)
        _tot['n'] = len(r)
        print(f'[tiling] 共 {len(r)} 块 (stage4 空间分块)', flush=True)
        return r

    _dt.prepare_tile_schedule = _prep

    _prog = {'i': 0, 't0': time.time()}
    _orig_tile = decoder._decode_one_tile

    def _tile(*a, **kw):
        _prog['i'] += 1
        k, n = _prog['i'], max(_tot['n'], 1)
        t = time.time()
        print(f'  [块 {k}/{n}] 开始 (已用 {t - _prog["t0"]:.0f}s'
              + (f', 上块 {(t - _prog["t0"]) / (k - 1):.0f}s, 预计剩 {(t - _prog["t0"]) / (k - 1) * (n - k + 1):.0f}s' if k > 1 else '')
              + ')', flush=True)
        r = _orig_tile(*a, **kw)
        print(f'  [块 {k}/{n}] 完成 {time.time() - t:.0f}s', flush=True)
        return r

    decoder._decode_one_tile = _tile
    _orig_grp = decoder._decode_temporal_group_isolated

    def _grp(tiles, *a, **kw):
        _tot['grp'] += 1
        print(f'[时间组 {_tot["grp"]}] {len(tiles)} 块', flush=True)
        return _orig_grp(tiles, *a, **kw)

    decoder._decode_temporal_group_isolated = _grp
    # ⚠️ 2K 必须开 tiling: 一次性解码 1408×2560×33帧 需 23GB 显存 → 12G 卡必爆 (2026-09-11)
    try:
        from ltx_core.tiling import TileSizeConfig, DimensionSizeConfig
        # 只分空间(768×768/重叠64), 时间轴不分块 — 短视频(33帧)默认时间重叠(24<40)不合规
        _tc = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=0, overlap=0),        # 时间轴不分块(重叠要求40帧)
            height=DimensionSizeConfig(tile_size=1024, overlap=160),    # 空间重叠须≥160(VAE感受野)
            width=DimensionSizeConfig(tile_size=1024, overlap=160),
        )
        print('tiling: 空间 1024x1024/160, 时间轴不分块', flush=True)
    except Exception as e:
        _tc = None
        print('tiling 不可用, 退回整图解码:', str(e)[:80], flush=True)
    with torch.inference_mode():
        chunks = []
        for ch in decoder.decode_video(latent, tiling_config=_tc):
            chunks.append(ch)
        frames = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        print(f'解码块数 {len(chunks)}', flush=True)
    # frames: [f,h,w,c] in [0,1]
    print(f'decode 完成 {time.time()-t1:.0f}s 帧 {tuple(frames.shape)}', flush=True)

    # 保存帧 png + mp4
    import imageio
    os.makedirs(r'D:\ai\output\ab_test', exist_ok=True)
    frames_np = (frames.float().cpu().numpy() * 255).clip(0, 255).astype('uint8')
    base = os.path.splitext(os.path.basename(latent_path))[0]
    out_mp4 = r'D:\ai\output\ab_test' + f'\\{base}.mp4'
    imageio.mimsave(out_mp4, frames_np, fps=24)
    print(f'[完成] mp4 保存 {out_mp4} 总耗时 {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else r'D:\ai\output\ab_test\lat_1024_512x512_f9.pt'
    main(p)
