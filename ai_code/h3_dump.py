# -*- coding: utf-8 -*-
"""H3 映射核对（零内存开销版）
   · 用 meta device 实例化 MiniMaxH3Transformer3DModel ⇒ 不占内存/显存
   · 打印 diffusers 侧真实参数名与 shape，并与 checkpoint 的键逐一对齐
   · 退出时强制归还内存（gc + empty_cache + os._exit）
"""
import os, sys, json, gc, time
sys.path.insert(0, r"D:\ai\py_h3")          # 隔离的 diffusers 0.41.0
sys.path.insert(0, r"D:\ai")
import torch

H3 = r"D:\ai\models\MiniMax-H3\Ref2VA"
cfg = json.load(open(os.path.join(H3, "transformer", "config.json"), encoding="utf-8"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "transformers_version"}
# config.json 里叫 token_refiner_num_layers，类里叫 num_refiner_layers ⇒ 做映射
RENAME = {"token_refiner_num_layers": "num_refiner_layers", "ffn_hidden_size": "ffn_dim"}
for a, b in RENAME.items():
    if a in cfg:
        cfg[b] = cfg.pop(a)
print("[dump] config:", {k: cfg[k] for k in list(cfg)[:8]}, flush=True)

import inspect
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

# ★ 只传类真正接受的参数：未知键自动改名/丢弃（config.json 与类命名不完全一致）
ACCEPT = set(inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters) - {"self"}
ALT = {
    "latents_dim": "in_channels", "audio_latents_dim": "audio_in_channels",
    "time_embed_hidden_size": "time_embed_hidden_dim", "rope_inv_freq_len": "rope_freq_dim",
    "timestep_input_dim": "freq_dim",
}
_fixed, _drop = {}, []
for k, v in cfg.items():
    t = k if k in ACCEPT else ALT.get(k)
    if t and t in ACCEPT:
        _fixed[t] = v
    else:
        _drop.append(k)
cfg = _fixed
print("[dump] 传入参数:", sorted(cfg.keys()), flush=True)
print("[dump] 丢弃（类不需要/已弃用）:", _drop, flush=True)
with torch.device("meta"):                     # ★ 零内存：只建结构，不要数据
    m = MiniMaxH3Transformer3DModel(**cfg)
print("[dump] 模型实例化 OK（meta device ✅ 不占内存）", flush=True)

names = [(n, tuple(p.shape)) for n, p in m.named_parameters()]
print("[dump] 参数总数:", len(names), flush=True)
b0 = [(n, s) for n, s in names if n.startswith("transformer_blocks.0.")]
print("[dump] === transformer_blocks.0 的参数（与 checkpoint 对齐核对）===", flush=True)
for n, s in b0:
    print("   %-52s %s" % (n.replace("transformer_blocks.0.", ""), s), flush=True)

# 与 checkpoint 的键对一遍
idx = json.load(open(os.path.join(H3, "transformer", "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
ck = sorted(k.split(".", 2)[2] for k in idx if k.startswith("blocks.0."))
print("[dump] checkpoint blocks.0 的键:", ck, flush=True)

def clean_exit(code=0):
    try:
        del m
    except Exception:
        pass
    gc.collect(); gc.collect()
    try:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    print("[dump] 已归还内存/显存（gc×2 + empty_cache ✅）", flush=True)
    sys.stdout.flush()
    os._exit(code)                              # ★ 强制退出，杜绝残留

clean_exit(0)
