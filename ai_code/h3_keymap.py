import os, sys, json, inspect, torch
sys.path.insert(0, r"D:\ai\py_h3"); sys.path.insert(0, r"D:\ai")
import h3_3tier_v2 as tier
idx = tier.H3LayerIndex()
print("=== checkpoint 里非层权重（others, %d 个）===" % len(idx.other))
for n, sp in idx.other: print("   ", n)
print("=== diffusers DiT 小件的参数名 ===")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
cfg = json.load(open(r"D:\ai\models\MiniMax-H3\Ref2VA\transformer\config.json", encoding="utf-8"))
ACC = set(inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters) - {"self"}
ALT = {"latents_dim":"in_channels","audio_latents_dim":"audio_in_channels","time_embed_hidden_size":"time_embed_hidden_dim","rope_inv_freq_len":"rope_freq_dim","timestep_input_dim":"freq_dim","token_refiner_num_layers":"num_refiner_layers","ffn_hidden_size":"ffn_dim"}
fx = {}
for k, v in cfg.items():
    if k.startswith("_") or k == "transformers_version": continue
    t = k if k in ACC else ALT.get(k)
    if t and t in ACC: fx[t] = v
with torch.device("meta"):
    m = MiniMaxH3Transformer3DModel(**fx)
for n, p in m.named_parameters():
    if not n.startswith("transformer_blocks."):
        print("   %-60s %s" % (n, tuple(p.shape)))
os._exit(0)