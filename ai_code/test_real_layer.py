"""
真实 HF forward: 逐层加载 stacked expert 权重
关键: HF state_dict 的 expert 是 stacked [512, ...], 和磁盘完全一致
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from non_expert_streamer import NonExpertStreamer
from expert_reader import SafeExpertReader
from bf16_compat import BF16_DTYPE

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'

# Build layer from config
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig
)
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = Qwen3_5MoeTextConfig(**config.text_config.to_dict())

def load_layer(layer_id, streamer, reader, activate_experts=10):
    """Build and load one layer from SSD via streamer + reader"""
    layer = Qwen3_5MoeDecoderLayer(tc, layer_idx=layer_id)

    # 1) Load non-expert weights
    ne = streamer.read_layer(layer_id)
    prefix = f"model.language_model.layers.{layer_id}."
    state = {}
    for name, arr in ne.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        # Skip expert weights (loaded separately)
        if 'mlp.experts.gate_up_proj' in key or 'mlp.experts.down_proj' in key:
            continue
        if arr.dtype == np.float32:
            state[key] = torch.from_numpy(arr.copy())
        else:
            state[key] = torch.from_numpy(arr.astype(np.uint16).view(np.int16)).view(torch.bfloat16)

    # 2) Build stacked expert weights
    # gate_up_proj: [512, 2048, 4096] bf16
    # down_proj:    [512, 4096, 1024] bf16
    # Fill unloaded experts with small random (not zeros) to avoid NaN from unused experts
    gu_stacked = torch.randn(512, 2048, 4096, dtype=torch.bfloat16) * 0.001
    dn_stacked = torch.randn(512, 4096, 1024, dtype=torch.bfloat16) * 0.001

    for eid in range(activate_experts):
        gu, dn = reader.read_expert(layer_id, eid)
        gu_t = torch.from_numpy(gu.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
        dn_t = torch.from_numpy(dn.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
        gu_stacked[eid] = gu_t
        dn_stacked[eid] = dn_t

    state['mlp.experts.gate_up_proj'] = gu_stacked
    state['mlp.experts.down_proj'] = dn_stacked

    layer.load_state_dict(state, strict=False)
    return layer.to("cuda", dtype=torch.bfloat16)

def _np_to_torch(arr):
    if arr.dtype == np.float32: return arr.copy()
    return torch.from_numpy(arr.astype(np.uint16).view(np.int16)).view(torch.bfloat16)

print("[Init] Streamer + Reader...")
streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

# Test layer 3 (full_attention)
print("\n=== Layer 3 (full_attention) ===")
layer = load_layer(3, streamer, reader)
hidden = torch.randn(1, 8, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
position_ids = torch.arange(0, 8, device="cuda").unsqueeze(0)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
rotary = Qwen3_5MoeTextRotaryEmbedding(config=tc).to("cuda")
pos_emb = rotary(hidden, position_ids)

with torch.no_grad():
    t0 = time.perf_counter()
    out = layer(hidden, position_embeddings=pos_emb, position_ids=position_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
result = out.float().cpu().numpy()
print(f"  Time={elapsed*1000:.0f}ms shape={result.shape} mean={result.mean():.6f} NaN={np.any(np.isnan(result))}")

del layer; torch.cuda.empty_cache()

# Test layer 0 (linear_attention)
print("\n=== Layer 0 (linear_attention) ===")
layer0 = load_layer(0, streamer, reader)
with torch.no_grad():
    t0 = time.perf_counter()
    out = layer0(hidden, position_embeddings=pos_emb, position_ids=position_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
result = out.float().cpu().numpy()
print(f"  Time={elapsed*1000:.0f}ms shape={result.shape} mean={result.mean():.6f} NaN={np.any(np.isnan(result))}")

del layer0; torch.cuda.empty_cache()

# Test: chain 4 layers (one full attention cycle)
print("\n=== Chain: layers 0-3 ===")
hidden2 = hidden.clone()
for lid in range(4):
    layer = load_layer(lid, streamer, reader)
    with torch.no_grad():
        hidden2 = layer(hidden2, position_embeddings=pos_emb, position_ids=position_ids)
    result = hidden2.float().cpu().numpy()
    print(f"  L{lid}: mean={result.mean():.6f} NaN={np.any(np.isnan(result))}")
    del layer; torch.cuda.empty_cache()

streamer.close(); reader.close()
print("\nDONE")
