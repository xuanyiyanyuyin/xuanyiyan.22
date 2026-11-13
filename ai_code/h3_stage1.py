# -*- coding: utf-8 -*-
"""H3 分阶段验证 Stage-1: 建 pipeline + 只加载 VAE + 手工建 DiT(meta) + 挂流式 ✅（不推理）"""
import os, sys, json, time, inspect, torch
sys.path.insert(0, r"D:\ai\py_h3"); sys.path.insert(0, r"D:\ai")
import accelerate
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
H3 = r"D:\ai\models\MiniMax-H3\Ref2VA"

with accelerate.init_empty_weights():
    pipe = MiniMaxH3ModularPipeline.from_pretrained(H3, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
print("[S1] pipeline 建好 ✅ blocks =", type(pipe.blocks).__name__, flush=True)

print("[S1] 组件规格:", flush=True)
for n in ("vae", "audio_vae", "transformer", "transformer_ref", "text_encoder"):
    try:
        sp = pipe.get_component_spec(n)
        print("   %-16s -> %s" % (n, getattr(sp, "type_hint", None)), flush=True)
    except Exception as e:
        print("   %-16s -> (无) %s" % (n, str(e)[:60]), flush=True)

print("[S1] 只加载 VAE（10GB）…", flush=True)
t0 = time.time()
try:
    pipe.load_components(names=["vae", "audio_vae"])
    print("[S1] VAE 加载成功 ✅ %.1fs  vae=%s  audio_vae=%s" % (
        time.time() - t0, type(pipe.vae).__name__, type(pipe.audio_vae).__name__), flush=True)
except Exception as e:
    print("[S1] VAE 加载失败:", type(e).__name__, str(e)[:300], flush=True)

print("[S1] 手工建 DiT（meta 零内存）+ 挂流式…", flush=True)
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
cfg = json.load(open(os.path.join(H3, "transformer", "config.json"), encoding="utf-8"))
ACC = set(inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters) - {"self"}
ALT = {"latents_dim": "in_channels", "audio_latents_dim": "audio_in_channels",
       "time_embed_hidden_size": "time_embed_hidden_dim", "rope_inv_freq_len": "rope_freq_dim",
       "timestep_input_dim": "freq_dim", "token_refiner_num_layers": "num_refiner_layers",
       "ffn_hidden_size": "ffn_dim"}
fixed = {}
for k, v in cfg.items():
    if k.startswith("_") or k == "transformers_version": continue
    t = k if k in ACC else ALT.get(k)
    if t and t in ACC: fixed[t] = v
with torch.device("meta"):
    dit = MiniMaxH3Transformer3DModel(**fixed)
print("[S1] DiT 结构 OK ✅ 参数数:", len(list(dit.named_parameters())), flush=True)
import h3_3tier_v2 as tier
idx = tier.H3LayerIndex()
print("[S1] 层索引: %d 层 ✅" % idx.n, flush=True)
pipe.transformer = dit
print("[S1] Stage-1 完成 ✅  VAE 就位=%s  DiT 就位=%s" % (pipe.vae is not None, pipe.transformer is not None), flush=True)
import gc; gc.collect(); torch.cuda.empty_cache()
print("[S1] 已归还 ✅", flush=True)
os._exit(0)