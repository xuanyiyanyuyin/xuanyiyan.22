# -*- coding: utf-8 -*-
"""MiniMax-H3 Ref2VA 生成入口 v2（2026-10-14 整段重写 · 干净版）"""
import argparse, os, sys, time, json, gc, inspect
import torch

H3_ROOT = r"D:\ai\models\MiniMax-H3\Ref2VA"
TURBO = {"8": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
         "4": r"D:\ai\models\MiniMax-H3\turbo\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"}

def build():
    import accelerate
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
    print("[g] 建 pipeline（init_empty_weights ⇒ 零内存）", flush=True)
    with accelerate.init_empty_weights():
        pipe = MiniMaxH3ModularPipeline.from_pretrained(H3_ROOT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    # 小件（processor/tokenizer/scheduler）正常加载
    try:
        pipe.load_components(names=["processor", "tokenizer", "scheduler", "audio_scheduler"])
        print("[g] processor/tokenizer/scheduler ✅", flush=True)
    except Exception as e:
        print("[g] load_components 部分失败:", str(e)[:150], flush=True)
    # 文本编码器（先试正常加载；失败再 offload）
    try:
        pipe.load_components(names=["text_encoder"])
        _bad = sum(1 for _, p in pipe.text_encoder.named_parameters() if p.device.type == "meta")
        print("[g] text_encoder ✅ meta 残留 %d" % _bad, flush=True)
        if _bad:
            raise RuntimeError("meta 残留")
    except Exception as e:
        print("[g] text_encoder 常规加载不行(%s) ⇒ 走 offload" % str(e)[:80], flush=True)
        from transformers import Qwen3VLForConditionalGeneration
        _od = os.path.join(H3_ROOT, "_offload_enc"); os.makedirs(_od, exist_ok=True)
        pipe.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(H3_ROOT, "text_encoder"), dtype=torch.bfloat16, device_map="auto",
            offload_folder=_od, offload_state_dict=True, low_cpu_mem_usage=True).eval()
        print("[g] text_encoder(offload) ✅", flush=True)
    # ★ VAE: 官方类（包式导入）+ fp32 + GPU
    sys.path.insert(0, H3_ROOT)
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
    _vdir = os.path.join(H3_ROOT, "video_vae")
    vae = MiniMaxH3VideoVAE.from_pretrained(_vdir).to(dtype=torch.float32)
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(os.path.join(H3_ROOT, "audio_vae")).to(dtype=torch.float32)
    _raw = json.load(open(os.path.join(_vdir, "config.json"), encoding="utf-8"))
    class _Cfg:
        def __init__(s, b, e): s._b, s._e = b, e
        def __getattr__(s, k):
            if k in object.__getattribute__(s, "_e"): return object.__getattribute__(s, "_e")[k]
            return getattr(object.__getattribute__(s, "_b"), k)
    vae.config = _Cfg(vae.config, {k: v for k, v in _raw.items() if not k.startswith("_")})
    vae.spatial_compression_ratio = int(_raw.get("vae_ratio", 16))
    vae.tokens_chunk_size = int(_raw.get("vae_latents_per_chunk", 5))
    # encode/decode 兼容（legacy: 不吃 return_dict；48 通道拆 24）
    _oe, _od2 = vae.encode, vae.decode
    class _Post:
        def __init__(s, t): s._t = t
        def _sp(s): return s._t[:, :24] if (s._t.dim() == 5 and s._t.shape[1] == 48) else s._t
        def sample(s, generator=None, **k): return s._sp()
        def mode(s): return s._sp()
        def __getitem__(s, i): return s if i == 0 else s._t[i]
        def __len__(s): return 1
        def __getattr__(s, k): return getattr(s._t, k)
    vae.encode = lambda x, *a, **k: (k.pop("return_dict", None), _oe(x.to(torch.float32) if torch.is_tensor(x) else x, *a, **k))[1]
    vae.decode = lambda x, *a, **k: _od2(x[:, :24] if (torch.is_tensor(x) and x.dim() == 5 and x.shape[1] == 48) else (x.to(torch.float32) if torch.is_tensor(x) else x), *a, **{kk: vv for kk, vv in k.items() if kk != "return_dict"})
    # Conv3d contiguous 包装
    import torch.nn as nn
    for _m in vae.modules():
        if isinstance(_m, nn.Conv3d):
            _o = _m.forward
            _m.forward = (lambda oo: (lambda z, *a, **k: oo(z.contiguous(), *a, **k)))(_o)
    try:
        vae.to("cuda"); avae.to("cuda")
        print("[g] VAE fp32 已上 GPU ✅（一次性，全程不搬 ✅）", flush=True)
    except Exception as e:
        print("[g] VAE 上 GPU 失败:", str(e)[:120], flush=True)
    pipe.vae, pipe.audio_vae = vae, avae
    print("[g] VAE ✅ video %dM audio %dM" % (sum(p.numel() for p in vae.parameters())/1e6, sum(p.numel() for p in avae.parameters())/1e6), flush=True)
    # ★ DiT: meta（bf16 默认）+ 流式
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    cfg = json.load(open(os.path.join(H3_ROOT, "transformer", "config.json"), encoding="utf-8"))
    ACC = set(inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters) - {"self"}
    ALT = {"latents_dim": "in_channels", "audio_latents_dim": "audio_in_channels", "time_embed_hidden_size": "time_embed_hidden_dim",
           "rope_inv_freq_len": "rope_freq_dim", "timestep_input_dim": "freq_dim", "token_refiner_num_layers": "num_refiner_layers", "ffn_hidden_size": "ffn_dim"}
    fx = {}
    for k, v in cfg.items():
        if k.startswith("_") or k == "transformers_version": continue
        t = k if k in ACC else ALT.get(k)
        if t and t in ACC: fx[t] = v
    _old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        dit = MiniMaxH3Transformer3DModel(**fx)
    torch.set_default_dtype(_old)
    pipe.transformer = dit
    print("[g] DiT(meta, bf16) ✅ %d 参数" % len(list(dit.named_parameters())), flush=True)
    return pipe, dit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--turbo", choices=["8", "4", "none"], default="none")
    ap.add_argument("--frames", type=int, default=124)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--resident", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompt = open(a.prompt_file, encoding="utf-8").read().strip()
    sys.path.insert(0, r"D:\ai\py_h3"); sys.path.insert(0, r"D:\ai")
    os.environ["H3_RESIDENT_LAYERS"] = str(a.resident)
    t0 = time.time()
    pipe, dit = build()
    import h3_3tier_v2 as tier
    idx = tier.H3LayerIndex()
    print("[g] 层索引 %d 层 ✅" % idx.n, flush=True)
    tier.patch_h3_forward(dit, index=idx, resident=a.resident, prefetch=True)
    if a.turbo != "none":
        from safetensors.torch import load_file
        sd = load_file(TURBO[a.turbo])
        pipe.load_lora_into_transformer(sd, adapter_name="turbo", transformer=dit)
        print("[g] Turbo LoRA(%s) 已挂 ✅ %d 张量" % (a.turbo, len(sd)), flush=True)
    from PIL import Image
    img = Image.open(a.ref).convert("RGB")
    print("[g] 开始推理 steps=%d frames=%d %dx%d img=%s（构建耗时 %.0fs）" % (a.steps, a.frames, a.width, a.height, img.size, time.time()-t0), flush=True)
    import numpy as np
    res = pipe(prompt=prompt, image=img, num_frames=a.frames, height=a.height, width=a.width,
               num_inference_steps=a.steps, generator=torch.Generator("cpu").manual_seed(42), output_type="np")
    fr = res.frames[0] if hasattr(res, "frames") else res[0]
    print("[g] ★ 推理完成 %.0fs 帧 shape=%s" % (time.time()-t0, getattr(fr, "shape", None)), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    try:
        import imageio.v3 as iio
        arr = np.clip(fr, 0.0, 1.0) if float(np.max(fr)) <= 1.5 else np.clip(fr, 0, 255)
        iio.imwrite(a.out, (arr*255 if float(np.max(arr)) <= 1.5 else arr).astype("uint8"), fps=24)
        print("[g] ★★ 已写出 " + a.out, flush=True)
    except Exception as e:
        print("[g] 写视频失败:", str(e)[:150], flush=True)
    return True

if __name__ == "__main__":
    import traceback
    ok = False
    try:
        ok = main()
    except BaseException as e:
        print("[g] 异常:", type(e).__name__, str(e)[:400], flush=True)
        traceback.print_exc()
    finally:
        gc.collect(); gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0 if ok else 1)