"""
自定义 MoE forward: 拆掉 stacked tensor, 逐个 expert 流式计算。
Router 常驻 GPU(4MB), expert 按需从 SSD→RAM→GPU 单次流转。
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from non_expert_streamer import NonExpertStreamer
from expert_reader import SafeExpertReader
from bf16_compat import BF16_DTYPE

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
DEVICE = "cuda"

from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig, Qwen3_5MoeRMSNorm
)

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = Qwen3_5MoeTextConfig(**config.text_config.to_dict())

streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

# ═══════════════════════════════════════════
# Custom MoE forward: 逐个 expert 流式
# ═══════════════════════════════════════════

def custom_moe_forward(hidden, router_weight, layer_id, reader,
                       shared_expert_gate, shared_gate, shared_up, shared_down,
                       top_k=10):
    """
    hidden: [batch, seq, 4096] on GPU
    router_weight: [512, 4096] on GPU (small, stays)
    layer_id: which layer
    reader: SafeExpertReader for loading experts from SSD

    Flow:
      1. Router: hidden @ router_weight^T → top-k indices
      2. For each of top-k experts:
         a. pread gate_up [16MB] + down [8MB] from SSD
         b. gate_up → GPU → split gate/up → SiLU(gate)*up @ down^T → partial output
         c. free expert weights from GPU
      3. Sum partial outputs + shared expert → final
    """
    batch, seq, hidden_dim = hidden.shape
    flat = hidden.reshape(-1, hidden_dim)  # [batch*seq, 4096]

    # ── Router ──
    router_logits = flat @ router_weight.T  # [N, 512]
    router_probs = torch.softmax(router_logits, dim=-1)
    _, top_indices = torch.topk(router_logits, k=top_k, dim=-1)  # [N, top_k]

    # ── Expert computation (serial, per-expert) ──
    expert_output = torch.zeros_like(flat)

    for e_rank in range(top_k):  # e_rank = 0..9 (position in top-k)
        expert_ids = top_indices[:, e_rank]  # [N]
        weights = router_probs[torch.arange(flat.shape[0]), expert_ids]  # [N]

        # Group same-expert tokens for efficiency
        unique_eids = expert_ids.unique().tolist()

        for eid in unique_eids:
            mask = expert_ids == eid
            if not mask.any():
                continue

            h_sub = flat[mask]  # tokens routed to this expert
            w = weights[mask].unsqueeze(-1)  # [n_sub, 1]

            # Load expert from SSD → numpy → torch GPU
            gu_np, dn_np = reader.read_expert(layer_id, eid)
            gu = torch.from_numpy(gu_np.astype(np.uint16).view(np.int16)).view(torch.bfloat16).to(DEVICE)
            dn = torch.from_numpy(dn_np.astype(np.uint16).view(np.int16)).view(torch.bfloat16).to(DEVICE)

            # SwiGLU: gate [1024,4096], up [1024,4096], down [4096,1024]
            gate_w = gu[:1024, :]  # [1024, 4096]
            up_w   = gu[1024:, :]  # [1024, 4096]

            gate = torch.nn.functional.silu(h_sub @ gate_w.T)  # [n_sub, 1024]
            up   = h_sub @ up_w.T                               # [n_sub, 1024]
            out  = (gate * up) @ dn.T                           # [n_sub, 4096]
            out  = out * w                                      # router weight

            expert_output[mask] += out

            # Free expert from GPU immediately
            del gu, dn, gate_w, up_w, gate, up, out, h_sub, w
            torch.cuda.empty_cache() if e_rank % 3 == 0 else None

    # ── Shared Expert ──
    gate_val = torch.sigmoid(flat @ shared_expert_gate.T)
    shared_out = torch.nn.functional.silu(flat @ shared_gate.T) * (flat @ shared_up.T)
    shared_out = shared_out @ shared_down.T
    expert_output = expert_output + gate_val * shared_out

    return expert_output.reshape(batch, seq, hidden_dim)


# ═══════════════════════════════════════════
# Test: single layer with custom MoE
# ═══════════════════════════════════════════

print("=== Custom MoE Forward Test (Layer 3) ===")

# Build minimal layer skeleton on GPU
layer = Qwen3_5MoeDecoderLayer(tc, layer_idx=3)

# Load only non-expert + shared expert weights (exclude routed experts)
ne = streamer.read_layer(3)
prefix = 'model.language_model.layers.3.'
sd = layer.state_dict()

for name, arr in ne.items():
    if not name.startswith(prefix): continue
    key = name[len(prefix):]
    if 'mlp.experts.gate_up_proj' in key or 'mlp.experts.down_proj' in key:
        continue  # routed experts handled by custom forward
    if key in sd:
        if arr.dtype == np.float32:
            sd[key].copy_(torch.from_numpy(arr.copy()))
        else:
            sd[key].copy_(torch.from_numpy(arr.astype(np.uint16).view(np.int16)).view(torch.bfloat16))

# Move only specific non-expert parts to GPU (NOT the whole layer)
# The whole layer would move 12.8GB of empty stacked expert tensors!
input_norm  = layer.input_layernorm.to(DEVICE, dtype=torch.bfloat16)
post_norm   = layer.post_attention_layernorm.to(DEVICE, dtype=torch.bfloat16)
self_attn   = layer.self_attn.to(DEVICE, dtype=torch.bfloat16)
print(f"Non-expert VRAM: {torch.cuda.memory_allocated()/1024**3:.3f}GB")

# Router + Shared expert — small, keep on GPU
router_w     = sd['mlp.gate.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_gate_w = sd['mlp.shared_expert_gate.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_gate   = sd['mlp.shared_expert.gate_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_up     = sd['mlp.shared_expert.up_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_down   = sd['mlp.shared_expert.down_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
print(f"+ Router/Shared VRAM: {torch.cuda.memory_allocated()/1024**3:.3f}GB")

# Input
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
hidden = torch.randn(1, 8, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
pos_ids = torch.arange(0, 8, device=DEVICE).unsqueeze(0)
rotary = Qwen3_5MoeTextRotaryEmbedding(config=tc).to(DEVICE)
pos_emb = rotary(hidden, pos_ids)

# ── Forward: RMSNorm + Attention (HF) + RMSNorm + Custom MoE + residual ──
t0 = time.perf_counter()
with torch.no_grad():
    # Attention block (HF native, non-expert weights already on GPU)
    residual = hidden.clone()
    hidden_norm = layer.input_layernorm(hidden)
    attn_out, _ = layer.self_attn(hidden_norm, position_embeddings=pos_emb,
                                   position_ids=pos_ids)
    hidden = residual + attn_out

    # MoE block via custom forward
    residual2 = hidden.clone()
    hidden_norm2 = layer.post_attention_layernorm(hidden)
    moe_out = custom_moe_forward(
        hidden_norm2, router_w, 3, reader,
        shared_gate_w, shared_gate, shared_up, shared_down
    )
    hidden = residual2 + moe_out

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

r = hidden.float().cpu().numpy()
print(f"\n=== RESULT ===")
print(f"Time: {elapsed*1000:.0f}ms")
print(f"Shape: {r.shape}  Mean: {r.mean():.6f}  Std: {r.std():.6f}")
print(f"NaN: {np.any(np.isnan(r))}  Inf: {np.any(np.isinf(r))}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
print(f"VRAM current: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

streamer.close(); reader.close()
torch.cuda.empty_cache()
print("\nDONE")
