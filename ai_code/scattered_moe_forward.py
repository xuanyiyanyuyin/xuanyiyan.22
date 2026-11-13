"""
ScatteredMoE: 替代 HF stacked index_select。
GPU 只放 10 个 expert (~250MB), 不是 512 个 (12.8GB)。
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from bf16_compat import BF16_DTYPE

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
DEVICE = "cuda"

from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig, Qwen3_5MoeTextRotaryEmbedding
)

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = Qwen3_5MoeTextConfig(**config.text_config.to_dict())

streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

class ScatteredMoEForward:
    """GPU 上只放 k 个 expert 的权重 buffer, 按需加载"""

    def __init__(self, k=10, intermediate=2048, hidden=4096):
        self.k = k
        # Small GPU buffers: only k experts, not 512
        self.gate_up_buf = torch.zeros(k, intermediate, hidden,
                                        dtype=torch.bfloat16, device=DEVICE)
        self.down_buf = torch.zeros(k, hidden, intermediate // 2,
                                     dtype=torch.bfloat16, device=DEVICE)
        print(f"[ScatteredMoE] GPU buffer: {self.gate_up_buf.nbytes/1024**2:.0f}MB "
              f"({k}/{512} experts)")

    def forward(self, hidden_flat, router_weight, layer_id, reader,
                shared_gate_w, shared_gate, shared_up, shared_down):
        """
        hidden_flat: [N, 4096] on GPU
        router_weight: [512, 4096] on GPU
        """
        N = hidden_flat.shape[0]

        # ── Router ──
        logits = hidden_flat @ router_weight.T  # [N, 512]
        probs = torch.softmax(logits, dim=-1)
        topk_weights, topk_ids = torch.topk(probs, k=self.k, dim=-1)

        # ── Collect unique experts needed ──
        unique_ids = topk_ids.unique().tolist()
        assert len(unique_ids) <= self.k, f"Need {len(unique_ids)} slots but only {self.k}"

        # ── Load experts from SSD to GPU buffer ──
        id_to_slot = {}
        for slot, eid in enumerate(unique_ids):
            gu_np, dn_np = reader.read_expert(layer_id, eid)
            gu_t = torch.from_numpy(gu_np.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
            dn_t = torch.from_numpy(dn_np.astype(np.uint16).view(np.int16)).view(torch.bfloat16)
            self.gate_up_buf[slot].copy_(gu_t.to(DEVICE, non_blocking=True))
            self.down_buf[slot].copy_(dn_t.to(DEVICE, non_blocking=True))
            id_to_slot[eid] = slot
        torch.cuda.synchronize()

        # ── Scattered compute: per-expert group ──
        output = torch.zeros_like(hidden_flat)

        for slot, eid in enumerate(unique_ids):
            mask = (topk_ids == eid).any(dim=-1)  # [N], which tokens hit this expert
            if not mask.any():
                continue

            tokens = hidden_flat[mask]  # [n, 4096]
            gu_w = self.gate_up_buf[slot]  # [2048, 4096]
            dn_w = self.down_buf[slot]     # [4096, 1024]

            # SwiGLU: gate [1024,4096] + up [1024,4096] → SiLU(gate)*up → down [4096,1024]
            gate_w = gu_w[:1024, :]
            up_w   = gu_w[1024:, :]
            gate = torch.nn.functional.silu(tokens @ gate_w.T)  # [n, 1024]
            up   = tokens @ up_w.T                              # [n, 1024]
            gated = gate * up                                   # [n, 1024]
            out   = gated @ dn_w.T                              # [n, 4096]

            # Router weights
            token_weights = topk_weights[mask]      # [n, k]
            expert_hit = (topk_ids[mask] == eid)    # [n, k]
            w = (token_weights * expert_hit).sum(dim=-1, keepdim=True)  # [n, 1]

            output[mask] += out * w

        # ── Shared Expert ──
        gate_val = torch.sigmoid(hidden_flat @ shared_gate_w.T)
        s_gate = torch.nn.functional.silu(hidden_flat @ shared_gate.T)
        s_up   = hidden_flat @ shared_up.T
        s_out  = (s_gate * s_up) @ shared_down.T
        output = output + gate_val * s_out

        return output


# ═══════════════════════════════════════════
# Build layer
# ═══════════════════════════════════════════
print("=== Scattered MoE Forward ===")

layer = Qwen3_5MoeDecoderLayer(tc, layer_idx=3)
ne = streamer.read_layer(3)
prefix = 'model.language_model.layers.3.'
sd = layer.state_dict()

for name, arr in ne.items():
    if not name.startswith(prefix): continue
    key = name[len(prefix):]
    if 'mlp.experts.gate_up' in key or 'mlp.experts.down_proj' in key: continue
    if key in sd:
        if arr.dtype == np.float32:
            sd[key].copy_(torch.from_numpy(arr.copy()))
        else:
            sd[key].copy_(torch.from_numpy(arr.astype(np.uint16).view(np.int16)).view(torch.bfloat16))

# Move only non-expert parts to GPU
input_norm = layer.input_layernorm.to(DEVICE, dtype=torch.bfloat16)
post_norm  = layer.post_attention_layernorm.to(DEVICE, dtype=torch.bfloat16)
self_attn  = layer.self_attn.to(DEVICE, dtype=torch.bfloat16)
print(f"Attention VRAM: {torch.cuda.memory_allocated()/1024**3:.3f}GB")

# Router + Shared
router_w      = sd['mlp.gate.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_gate_w = sd['mlp.shared_expert_gate.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_gate   = sd['mlp.shared_expert.gate_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_up     = sd['mlp.shared_expert.up_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
shared_down   = sd['mlp.shared_expert.down_proj.weight'].to(DEVICE, dtype=torch.bfloat16)
print(f"+ MoE meta VRAM: {torch.cuda.memory_allocated()/1024**3:.3f}GB")

# Create scattered MoE
moe = ScatteredMoEForward(k=10)

# Input
hidden = torch.randn(1, 8, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
pos_ids = torch.arange(0, 8, device=DEVICE).unsqueeze(0)
rotary = Qwen3_5MoeTextRotaryEmbedding(config=tc).to(DEVICE)
pos_emb = rotary(hidden, pos_ids)

# ── Forward ──
t0 = time.perf_counter()
with torch.no_grad():
    # Attention
    residual = hidden.clone()
    hn = input_norm(hidden)
    attn_out, _ = self_attn(hn, attention_mask=None,
                             position_embeddings=pos_emb, position_ids=pos_ids)
    hidden = residual + attn_out

    # Custom MoE
    residual2 = hidden.clone()
    flat = post_norm(hidden).reshape(-1, 4096)
    moe_out = moe.forward(flat, router_w, 3, reader,
                          shared_gate_w, shared_gate, shared_up, shared_down)
    hidden = residual2 + moe_out.reshape(1, 8, 4096)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

r = hidden.float().cpu().numpy()
print(f"\n=== RESULT ===")
print(f"Time: {elapsed*1000:.0f}ms")
print(f"Mean: {r.mean():.6f}  Std: {r.std():.6f}")
print(f"NaN: {np.any(np.isnan(r))}  Inf: {np.any(np.isinf(r))}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

streamer.close(); reader.close()
torch.cuda.empty_cache()
print("DONE")
