"""
真实 HF Qwen3.5 单层 forward 验证。
用 transformers 构建层结构，streamer 加载权重，CUDA 计算。
"""
import sys; sys.path.insert(0, r'D:\ai')
import time, numpy as np, torch
from non_expert_streamer import NonExpertStreamer
from expert_reader import SafeExpertReader
from bf16_compat import BF16_DTYPE

# 1) Build single layer from config
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig, Qwen3_5MoeRMSNorm
)
import json

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = config.text_config.to_dict()
text_cfg = Qwen3_5MoeTextConfig(**tc)

print(f"[Config] hidden={text_cfg.hidden_size}, layers={text_cfg.num_hidden_layers}, "
      f"experts={text_cfg.num_experts}, topk={text_cfg.num_experts_per_tok}")
print(f"[Layers] linear_attention={text_cfg.layer_types.count('linear_attention')}, "
      f"full_attention={text_cfg.layer_types.count('full_attention')}")

# 2) Create decoder layer object (no weights loaded yet)
print("\n[Build] Creating layer 3 (full_attention) and layer 0 (linear_attention)...")

# Need to use torch's load with assign=True to map weights
# Create layers with config
layer3 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=3)
layer0 = Qwen3_5MoeDecoderLayer(text_cfg, layer_idx=0)

# 3) Load weights from streamer and assign
streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

def assign_weights(module, layer_id, weights_dict):
    """Assign numpy weights to HF module using state_dict keys"""
    prefix = f"model.language_model.layers.{layer_id}."
    state = {}
    for name, arr in weights_dict.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]  # strip prefix
        # Skip routed experts (they're loaded separately)
        if 'mlp.experts.' in key and 'shared_expert' not in key and '.gate.' not in key:
            if 'experts.0.' in key or 'experts.1.' in key:  # skip non-expert-0/1
                continue
        state[key] = _np_to_torch(arr)

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        routed = [k for k in missing if 'experts' in k]
        other = [k for k in missing if 'experts' not in k]
        if len(routed) > 0:
            print(f"  [Skip] {len(routed)} routed expert weights (loaded separately)")
        if other:
            print(f"  [Missing] {len(other)} non-expert keys: {other[:3]}")

def _np_to_torch(arr):
    if arr.dtype == np.float32:
        return arr.copy()
    elif arr.dtype in [np.uint16, BF16_DTYPE]:
        return torch.from_numpy(arr.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
    else:
        return arr.astype(np.float32)

# Load and assign for layer 3
print("\n[Load] Layer 3 weights...")
ne3 = streamer.read_layer(3)
assign_weights(layer3, 3, ne3)
layer3 = layer3.to("cuda", dtype=torch.bfloat16)

# Load expert weights for MoE
# The MoE uses expert weights in the SparseMoeBlock
# We load 10 experts and set them
print("[Load] Layer 3 experts...")
expert_state = {}
for eid in range(10):
    gu, dn = reader.read_expert(3, eid)
    # gate_up: [2048, 4096] -> split into gate [1024, 4096] and up [1024, 4096]
    gate = gu[:1024, :]
    up = gu[1024:, :]
    expert_state[f"mlp.experts.experts.{eid}.gate_proj.weight"] = torch.from_numpy(gate.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
    expert_state[f"mlp.experts.experts.{eid}.up_proj.weight"] = torch.from_numpy(up.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
    expert_state[f"mlp.experts.experts.{eid}.down_proj.weight"] = torch.from_numpy(dn.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
layer3.load_state_dict(expert_state, strict=False)

# 4) Forward test
print("\n[Forward] Layer 3 (full_attention)...")
hidden = torch.randn(1, 8, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
position_ids = torch.arange(0, 8, device="cuda").unsqueeze(0)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
rotary = Qwen3_5MoeTextRotaryEmbedding(config=text_cfg)
rotary = rotary.to("cuda")
pos_emb = rotary(hidden, position_ids)

with torch.no_grad():
    t0 = time.perf_counter()
    output = layer3(hidden, position_embeddings=pos_emb, position_ids=position_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

result = output.float().cpu().numpy()
print(f"  shape={result.shape} mean={result.mean():.6f} std={result.std():.6f} "
      f"NaN={np.any(np.isnan(result))} Inf={np.any(np.isinf(result))}")
print(f"  Time: {elapsed*1000:.1f}ms")

# Also test layer 0 (linear_attention)
print("\n[Load] Layer 0 weights...")
ne0 = streamer.read_layer(0)
assign_weights(layer0, 0, ne0)
layer0 = layer0.to("cuda", dtype=torch.bfloat16)

for eid in range(10):
    gu, dn = reader.read_expert(0, eid)
    gate = gu[:1024, :]; up = gu[1024:, :]
    expert_state[f"mlp.experts.experts.{eid}.gate_proj.weight"] = torch.from_numpy(gate.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
    expert_state[f"mlp.experts.experts.{eid}.up_proj.weight"] = torch.from_numpy(up.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
    expert_state[f"mlp.experts.experts.{eid}.down_proj.weight"] = torch.from_numpy(dn.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
layer0.load_state_dict(expert_state, strict=False)

print("[Forward] Layer 0 (linear_attention)...")
with torch.no_grad():
    t0 = time.perf_counter()
    output = layer0(hidden, position_embeddings=pos_emb, position_ids=position_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

result = output.float().cpu().numpy()
print(f"  shape={result.shape} mean={result.mean():.6f} std={result.std():.6f} "
      f"NaN={np.any(np.isnan(result))} Inf={np.any(np.isinf(result))}")
print(f"  Time: {elapsed*1000:.1f}ms")

streamer.close()
reader.close()
torch.cuda.empty_cache()
print("\nDONE")
