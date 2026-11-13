# -*- coding: utf-8 -*-
"""MiniMax-H3 Ref2VA 生成入口（参考图 -> 视频 + 音频）
   · 走隔离 diffusers 0.41.0（PYTHONPATH=D:\ai\py_h3）
   · DiT 50 层用 h3_3tier.patch_layer_loop 做逐层 pread 流式（12G 显存的关键）
   · Turbo LoRA（8 步 / 4 步）通过 PeftAdapterMixin 挂载
用法示例:
  set PYTHONPATH=D:\ai\py_h3
  python h3_gen.py --ref ref.png --prompt "..." --steps 8 --turbo 8 --out D:\ai\output\h3_test.mp4
"""
import argparse, os, sys, time, json
import torch

H3_ROOT = r"D:\ai\models\MiniMax-H3\Ref2VA"
TURBO = {
    "8": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "4": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None, help="从文件读提示词（避免命令行空格被拆）")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--turbo", choices=["8", "4", "none"], default="8")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resident", type=int, default=3, help="常驻 GPU 的 DiT 层数")
    ap.add_argument("--dump-names", action="store_true", help="只打印 diffusers block 参数名（核对映射用）")
    a = ap.parse_args()
    if a.prompt_file:
        a.prompt = open(a.prompt_file, encoding="utf-8").read().strip()
    if not a.prompt:
        raise SystemExit("必须给 --prompt 或 --prompt-file")

    os.environ["H3_RESIDENT_LAYERS"] = str(a.resident)
    sys.path.insert(0, r"D:\ai\py_h3")
    sys.path.insert(0, r"D:\ai")
    # ★ 必须从子模块精确导入（顶层导出的那个没有 default_blocks_name ⇒ 会报 no blocks）
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline  # noqa
    import h3_3tier_v2 as tier                                  # noqa

    t0 = time.time()
    print("[h3_gen] 加载 pipeline（DiT 用 meta 空权重，避免整份 61.7GB 进内存）…", flush=True)
    offload_dir = r"D:\ai\models\MiniMax-H3\_offload"
    os.makedirs(offload_dir, exist_ok=True)
    # ★ 关键: 在 accelerate 的 init_empty_weights 上下文里构建 ⇒ 全部模块都是 meta（零内存）
    import accelerate
    print("[h3_gen] 用 accelerate.init_empty_weights() 构建（全部 meta ✅ 零内存占用）", flush=True)
    with accelerate.init_empty_weights():
        pipe = MiniMaxH3ModularPipeline.from_pretrained(
            H3_ROOT,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    # ★ 先加载小件（processor/tokenizer 必须，否则 encoders.py 报 None.image_processor）
    try:
        pipe.load_components(names=["processor", "tokenizer"])
        print("[h3_gen] processor ✅ tokenizer ✅", flush=True)
    except Exception as e:
        print("[h3_gen] processor/tokenizer 加载失败:", str(e)[:200], flush=True)
    # ★ 文本编码器 Qwen3-VL（62GB）: 用 accelerate 磁盘 offload ⇒ 权重不全进内存 · 计算逐层上 GPU ✅
    OFFLOAD_ENCODER = True
    _odir = os.path.join(H3_ROOT, "_offload_enc")
    os.makedirs(_odir, exist_ok=True)
    try:
        if OFFLOAD_ENCODER:
            from transformers import Qwen3VLForConditionalGeneration
            _te = Qwen3VLForConditionalGeneration.from_pretrained(
                os.path.join(H3_ROOT, "text_encoder"),
                dtype=torch.bfloat16,
                device_map="auto",                 # ★ 自动分配: 一部分在 GPU、其余在 CPU/磁盘 ✅
                offload_folder=_odir,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
            )
            _te.eval()
            pipe.text_encoder = _te
            print("[h3_gen] text_encoder 已用 offload 加载 ✅ device_map=%s" % (
                getattr(_te, "hf_device_map", {}).get("", "auto") if hasattr(_te, "hf_device_map") else "auto"), flush=True)
        else:
            pipe.load_components(names=["text_encoder"])
        print("[h3_gen] text_encoder(Qwen3-VL) ✅", flush=True)
    except Exception as e:
        print("[h3_gen] text_encoder offload 失败，回落 load_components:", str(e)[:200], flush=True)
        try:
            pipe.load_components(names=["text_encoder"])
            print("[h3_gen] text_encoder ✅ (load_components)", flush=True)
        except Exception as e2:
            print("[h3_gen] text_encoder 全失败:", str(e2)[:150], flush=True)
    # ★ 调度器（视频 + 音频各一个 ✅）
    try:
        pipe.load_components(names=["scheduler", "audio_scheduler"])
        print("[h3_gen] scheduler ✅ audio_scheduler ✅ (load_components)", flush=True)
    except Exception as _e:
        print("[h3_gen] load_components 调度器失败，改手工:", str(_e)[:120], flush=True)
        try:
            from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
            pipe.scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(H3_ROOT, "scheduler"))
            pipe.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(H3_ROOT, "audio_scheduler"))
            print("[h3_gen] scheduler ✅ audio_scheduler ✅ (手工)", flush=True)
        except Exception as _e2:
            print("[h3_gen] 调度器手工加载也失败:", str(_e2)[:150], flush=True)
    # ★ 手工装配 VAE（不走 load_components：它对源格式子目录的解析有坑）
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    # ★ 用官方自带类读源格式（标准类会对不上键 ⇒ 544 个参数留 meta）
    import sys as _sys
    _vdir = os.path.join(H3_ROOT, "video_vae")
    _sys.path.insert(0, H3_ROOT)        # ★ 必须按【包】导入（官方源码用相对导入 from .attention ✅）
    try:
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE as _VVAE
        vae = _VVAE.from_pretrained(_vdir, dtype=torch.bfloat16)   # ★ bf16 conv3d 实测 CUDA 支持 ✅ 省一半显存
        # ★ 把源 config 的字段注入 vae.config（pipeline 需要 latents_mean/std 等 ✅）
        import json as _json
        class _CfgWrap:
            def __init__(self, base, extra):
                self._b, self._e = base, extra
            def __getattr__(self, k):
                if k in self._e:
                    return self._e[k]
                try:
                    return getattr(self._b, k)
                except Exception:
                    raise AttributeError(k)
        _raw = _json.load(open(os.path.join(_vdir, "config.json"), encoding="utf-8"))
        _extra = {k: v for k, v in _raw.items() if not k.startswith("_")}
        vae.config = _CfgWrap(vae.config, _extra)
        _inject_vae_cfg = True
        # ★ 兼容层: 官方 VAE 是 legacy 风格，encode/decode 不接受 return_dict 等 kwargs
        _oe, _od = vae.encode, vae.decode
        class _PostWrap:
            """让 legacy VAE 的 Tensor 输出表现得像 DiagonalGaussianDistribution ✅"""
            def __init__(self, t):
                self._t = t
            def _split(self):
                """legacy VAE 返回 mean+logvar 拼接 ⇒ 拆开取 mean（24 通道 ✅）"""
                import os as _o, json as _j
                _lc = 24
                try:
                    _c = _j.load(open(os.path.join(H3_ROOT, "video_vae", "config.json"), encoding="utf-8"))
                    _lc = int(_c.get("latent_channels", 24))
                except Exception:
                    pass
                t = self._t
                if t.shape[1] == 2 * _lc:
                    return t[:, :_lc]
                return t
            def sample(self, generator=None, **kw):
                return self._split()
            def mode(self):
                return self._split()
            def parameters(self):
                return self._t
            def __getitem__(self, i):
                return self if i == 0 else self._t[i]      # ★ pipeline 会写 [0] 再 .sample() ✅
            def __len__(self):
                return 1
            def __getattr__(self, k):
                return getattr(self._t, k)
    except Exception as _e3:
        print("[h3_gen] VAE 装配失败（回退标准类）:", str(_e3)[:200], flush=True)
        vae = AutoencoderKLMiniMaxH3.from_pretrained(_vdir, dtype=torch.bfloat16)

    # ★ meta 残留扫描（2026-10-14 加）：指名道姓找出还有 meta 的组件
    def _scan_meta(tag, mod):
        if mod is None:
            print("[meta] %s: None" % tag, flush=True); return 0
        badb = [n for n, b in mod.named_buffers() if b.device.type == "meta"]
        if badb:
            print("[meta] %s buffers meta %d: %s" % (tag, len(badb), badb[:4]), flush=True)
        bad = [n for n, p in mod.named_parameters() if p.device.type == "meta"]
        tot = sum(1 for _ in mod.named_parameters())
        print("[meta] %-16s meta %d / 共 %d  %s" % (tag, len(bad), tot, (bad[:4] if bad else "✅")), flush=True)
        return len(bad)
    print("[h3_gen] === meta 残留扫描 ===", flush=True)
    for _t, _m in (("transformer", pipe.transformer), ("video_vae", pipe.vae), ("audio_vae", pipe.audio_vae),
                   ("text_encoder", getattr(pipe, "text_encoder", None)),
                   ("transformer_ref", getattr(pipe, "transformer_ref", None))):
        _scan_meta(_t, _m)

    if a.dump_names:
        print("[h3_gen] block 参数名:", tier.dump_block_param_names(dit)[:20], flush=True)
        return

    idx = tier.H3LayerIndex()
    print("[h3_gen] DiT 层数=%d 常驻=%d（逐层磁盘 pread 流式）" % (idx.n, a.resident), flush=True)
    tier.patch_h3_forward(dit, index=idx, resident=a.resident, prefetch=True)

    if a.turbo != "none":
        p = TURBO[a.turbo]
        print("[h3_gen] 挂 Turbo LoRA(%s): %s" % (a.turbo, os.path.basename(p)), flush=True)
        from safetensors.torch import load_file
        sd = load_file(p)
        print("[h3_gen] LoRA 张量数:", len(sd), "例:", list(sd.keys())[:3], flush=True)
        done = False
        # ★ LoRA 在 pipe 上（不是 dit）
        for call in ("load_lora_into_transformer", "load_lora_weights"):
            fn = getattr(pipe, call, None)
            if fn is None:
                continue
            try:
                fn(sd, adapter_name="turbo", transformer=dit)
                print("[h3_gen] LoRA 已挂（pipe.%s）✅" % call, flush=True)
                done = True
                break
            except Exception as e:
                print("[h3_gen] pipe.%s 失败: %s" % (call, str(e)[:150]), flush=True)
        if not done:
            try:
                from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers as conv
                sd2 = conv(sd)
                pipe.load_lora_into_transformer(sd2, adapter_name="turbo", transformer=dit)
                print("[h3_gen] LoRA 已挂（转换后）✅", flush=True)
                done = True
            except Exception as e:
                print("[h3_gen] LoRA 转换挂载也失败:", str(e)[:200], flush=True)
        if not done:
            print("[h3_gen] ⚠️ 未挂 LoRA，将以默认步数跑（可能很慢）", flush=True)

    # ★ 预判②: scheduler 的 sigmas/timesteps 统一到 CPU（避免 step() 里设备混用 ✅）
    for _sn in ("scheduler", "audio_scheduler"):
        _s = getattr(pipe, _sn, None)
        if _s is None:
            continue
        for _attr in ("sigmas", "timesteps", "_step_index", "_begin_index"):
            try:
                _v2 = getattr(_s, _attr, None)
                if torch.is_tensor(_v2):
                    setattr(_s, _attr, _v2.cpu())
            except Exception:
                pass
    print("[h3_gen] scheduler sigmas/timesteps 已统一 CPU ✅（预判②）", flush=True)
    print("[h3_gen] 开始推理 steps=%d frames=%d %dx%d" % (a.steps, a.frames, a.width, a.height), flush=True)
    from PIL import Image as _Img
    _ref_img = _Img.open(a.ref).convert("RGB")      # ★ pipeline 要图像对象，不是路径字符串
    print("[h3_gen] 参考图已载入 %s" % (_ref_img.size,), flush=True)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    imgs = pipe(
        prompt=a.prompt,
        image=_ref_img,
        num_frames=a.frames,
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps,
        generator=g,
        output_type="np",
    )
    frames = imgs.frames[0] if hasattr(imgs, "frames") else imgs[0]
    print("[h3_gen] 推理完成 %.1fs  帧 shape=%s" % (time.time() - t0, getattr(frames, "shape", None)), flush=True)
    try:
        import imageio.v3 as iio
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        import numpy as _np
        _f = _np.clip(frames, 0.0, 1.0) if float(_np.max(frames)) <= 1.5 else _np.clip(frames, 0.0, 255.0)
        iio.imwrite(a.out, (_f * 255 if float(_np.max(_f)) <= 1.5 else _f).astype("uint8"), fps=24)   # ★ 预判④ clip ✅
        print("[h3_gen] 已写出", a.out, flush=True)
    except Exception as e:
        print("[h3_gen] 写视频失败（帧已在内存）:", str(e)[:150], flush=True)

def _cleanup():
    import gc
    gc.collect(); gc.collect()
    try:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    print("[h3_gen] 已归还内存/显存 ✅", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    import traceback
    ok = False
    try:
        main()
        ok = True
    except SystemExit as e:
        print("[h3_gen] SystemExit:", e.code, flush=True)
    except BaseException as e:
        print("[h3_gen] 异常:", type(e).__name__, str(e)[:500], flush=True)
        traceback.print_exc()
    finally:
        _cleanup()
        sys.stdout.flush(); sys.stderr.flush()
        if ok:
            os._exit(0)      # 成功才强制退出（归还虚拟内存 ✅）
        # 失败时正常退出，让异常信息能被看到 ✅
