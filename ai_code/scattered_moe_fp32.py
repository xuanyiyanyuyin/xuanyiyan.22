"""
ScatteredMoE: 全部 fp32，避开 bf16 转换坑。
GPU 只放 10 个 expert (~250MB fp32 = ~500MB), 不是 512 个。
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer

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


def bf16_bytes_to_fp32(raw_bytes: bytes) -> np.ndarray:
    """bfloat16 raw bytes → float32 numpy array"""
    u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32


def load_expert_fp32(reader, layer_id, eid):
    """Load expert weights as fp32 numpy, then torch fp32"""
    gu_np, dn_np = reader.read_expert(layer_id, eid)
    # reader returns uint16 numpy. Convert to fp32.
    gu_f32 = (gu_np.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    dn_f32 = (dn_np.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return gu_f32, dn_f32


class ScatteredMoEFP32:
    def __init__(self, k=10, intermediate=2048, hidden=4096, max_unique=32):
        self.k = k
        self.max_unique = max_unique
        self.gate_up_buf = torch.zeros(max_unique, intermediate, hidden, device=DEVICE)
        self.down_buf = torch.zeros(max_unique, hidden, intermediate//2, device=DEVICE)

    def forward(self, hidden_flat, router_weight, layer_id, reader,
                sgw, sg, su, sd):
        N = hidden_flat.shape[0]

        # Router
        logits = hidden_flat @ router_weight.T
        probs = torch.softmax(logits, dim=-1)
        topk_w, topk_ids = torch.topk(probs, k=self.k, dim=-1)
        unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.max_unique:
            unique_ids = unique_ids[:self.max_unique]

        # Load experts
        id_to_slot = {}
        for slot, eid in enumerate(unique_ids):
            gu_f32, dn_f32 = load_expert_fp32(reader, layer_id, eid)
            self.gate_up_buf[slot].copy_(torch.from_numpy(gu_f32).to(DEVICE))
            self.down_buf[slot].copy_(torch.from_numpy(dn_f32).to(DEVICE))
            id_to_slot[eid] = slot
        torch.cuda.synchronize()

        # Compute
        output = torch.zeros_like(hidden_flat)
        for slot, eid in enumerate(unique_ids):
            mask = (topk_ids == eid).any(dim=-1)
            if not mask.any(): continue
            tokens = hidden_flat[mask]
            gu_w = self.gate_up_buf[slot]
            dn_w = self.down_buf[slot]
            gate_w, up_w = gu_w[:1024], gu_w[1024:]
            gate = torch.nn.functional.silu(tokens @ gate_w.T)
            up = tokens @ up_w.T
            out = (gate * up) @ dn_w.T
            tw = topk_w[mask]; eh = (topk_ids[mask] == eid)
            w = (tw * eh).sum(dim=-1, keepdim=True)
            output[mask] += out * w

        # Shared expert
        gate_v = torch.sigmoid(hidden_flat @ sgw.T)
        s_gate = torch.nn.functional.silu(hidden_flat @ sg.T)
        s_up = hidden_flat @ su.T
        s_out = (s_gate * s_up) @ sd.T
        output = output + gate_v * s_out
        return output


# ═══════ Build & Test ═══════
print("=== Scattered MoE FP32 ===")
layer = Qwen3_5MoeDecoderLayer(tc, layer_idx=3)
ne = streamer.read_layer(3)
prefix = 'model.language_model.layers.3.'
sd = layer.state_dict()

for name, arr in ne.items():
    if not name.startswith(prefix): continue
    key = name[len(prefix):]
    if 'mlp.experts.gate_up' in key or 'mlp.experts.down_proj' in key: continue
    if key in sd:
        # np → torch (handle both fp32 and bf16 numpy)
        if arr.dtype == np.float32:
            sd[key].copy_(torch.from_numpy(arr.copy()))
        else:
            # bfloat16 via ml_dtypes: MUST use view(np.uint16) to preserve raw bytes
            f32 = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            sd[key].copy_(torch.from_numpy(f32.copy()))

input_norm = layer.input_layernorm.to(DEVICE)
post_norm = layer.post_attention_layernorm.to(DEVICE)
self_attn = layer.self_attn.to(DEVICE)

router_w = sd['mlp.gate.weight'].to(DEVICE)
sgw = sd['mlp.shared_expert_gate.weight'].to(DEVICE)
sg = sd['mlp.shared_expert.gate_proj.weight'].to(DEVICE)
su = sd['mlp.shared_expert.up_proj.weight'].to(DEVICE)
sd_ = sd['mlp.shared_expert.down_proj.weight'].to(DEVICE)
print(f"VRAM base: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

moe = ScatteredMoEFP32(k=10)

hidden = torch.randn(1, 8, 4096, device=DEVICE) * 0.02
pos_ids = torch.arange(0, 8, device=DEVICE).unsqueeze(0)
rotary = Qwen3_5MoeTextRotaryEmbedding(config=tc).to(DEVICE)
pos_emb = rotary(hidden, pos_ids)

t0 = time.perf_counter()
with torch.no_grad():
    residual = hidden.clone()
    hn = input_norm(hidden)
    attn_out, _ = self_attn(hn, attention_mask=None, position_embeddings=pos_emb, position_ids=pos_ids)
    hidden = residual + attn_out

    residual2 = hidden.clone()
    flat = post_norm(hidden).reshape(-1, 4096)
    moe_out = moe.forward(flat, router_w, 3, reader, sgw, sg, su, sd_)
    hidden = residual2 + moe_out.reshape(1, 8, 4096)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

r = hidden.cpu().numpy()
print(f"Time={elapsed*1000:.0f}ms Mean={r.mean():.6f} NaN={np.any(np.isnan(r))}")
print(f"VRAM peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
streamer.close(); reader.close()
print("DONE")
