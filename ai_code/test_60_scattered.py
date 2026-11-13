"""
60 层 Scattered MoE 全栈 forward。
每层: streamer 加载非专家 → HF Attention → 自定义 ScatteredMoE → 释放。
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from layer_utils import load_layer_types
sys.path.insert(0, r'D:\ai')

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
DEVICE = "cuda"

from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextConfig, Qwen3_5MoeTextRotaryEmbedding
)

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = Qwen3_5MoeTextConfig(**config.text_config.to_dict())
layer_types = load_layer_types(MODEL)

streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

# ── Shared MoE module (reuse across layers) ──
class ScatteredMoE:
    def __init__(self, max_unique=32, inter=2048, hidden=4096):
        self.gu_buf = torch.zeros(max_unique, inter, hidden, device=DEVICE)
        self.dn_buf = torch.zeros(max_unique, hidden, inter//2, device=DEVICE)
        self.max_unique = max_unique

    def forward(self, hidden_flat, router_w, layer_id,
                sgw, sg, su, sd):
        N = hidden_flat.shape[0]
        logits = hidden_flat @ router_w.T
        probs = torch.softmax(logits, dim=-1)
        topk_w, topk_ids = torch.topk(probs, k=10, dim=-1)
        unique_ids = topk_ids.unique().tolist()[:self.max_unique]
        id_to_slot = {}

        for slot, eid in enumerate(unique_ids):
            gu_np, dn_np = reader.read_expert(layer_id, eid)
            gu_f32 = (gu_np.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            dn_f32 = (dn_np.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            self.gu_buf[slot].copy_(torch.from_numpy(gu_f32).to(DEVICE))
            self.dn_buf[slot].copy_(torch.from_numpy(dn_f32).to(DEVICE))
            id_to_slot[eid] = slot
        torch.cuda.synchronize()

        output = torch.zeros_like(hidden_flat)
        for slot, eid in enumerate(unique_ids):
            mask = (topk_ids == eid).any(dim=-1)
            if not mask.any(): continue
            tokens = hidden_flat[mask]
            gu_w = self.gu_buf[slot]; dn_w = self.dn_buf[slot]
            gate_w, up_w = gu_w[:1024], gu_w[1024:]
            gate = torch.nn.functional.silu(tokens @ gate_w.T)
            up = tokens @ up_w.T
            out = (gate * up) @ dn_w.T
            tw = topk_w[mask]; eh = (topk_ids[mask] == eid)
            w = (tw * eh).sum(dim=-1, keepdim=True)
            output[mask] += out * w

        gate_v = torch.sigmoid(hidden_flat @ sgw.T)
        s_out = (torch.nn.functional.silu(hidden_flat @ sg.T) * (hidden_flat @ su.T)) @ sd.T
        output = output + gate_v * s_out
        return output

moe = ScatteredMoE()

# ── Monkey-patch: allocate tiny experts (1 instead of 512) ──
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
_orig_init = Qwen3_5MoeExperts.__init__
def _patched_init(self, config):
    config.num_experts = 1  # fake 1 expert to avoid 17GB allocation
    _orig_init(self, config)
    config.num_experts = 512  # restore
Qwen3_5MoeExperts.__init__ = _patched_init

# ── Build ONE template (reusing same object all 60 layers) ──
# Create with layer 0's type, switch block_type per iteration
layer = Qwen3_5MoeDecoderLayer(tc, layer_idx=0)
expert_keys_in_sd = {k for k in layer.state_dict() if 'experts' in k and 'shared' not in k}
print("Template allocated")

# ── Forward loop ──
print("=" * 60)
print("60-Layer Scattered MoE Forward")
print("=" * 60)

hidden = torch.randn(1, 8, 4096, device=DEVICE) * 0.02
rotary = Qwen3_5MoeTextRotaryEmbedding(config=tc).to(DEVICE)

times_ssd, times_attn, times_moe = [], [], []
nan_layers = []

for lid in range(60):
    # A) Load non-expert weights
    t1 = time.perf_counter()
    ne = streamer.read_layer(lid)
    t2 = time.perf_counter()

    # B) Reassign block type, zero non-expert, reload
    block_type = layer_types[lid]
    layer.block_type = block_type
    # Zero all non-expert tensors
    for k, v in layer.state_dict().items():
        if k not in expert_keys_in_sd:
            v.zero_()
    for k, v in layer.state_dict().items():
        if k not in expert_keys_in_sd:
            v.zero_()
    prefix = f'model.language_model.layers.{lid}.'

    layer_sd = layer.state_dict()
    for name, arr in ne.items():
        if not name.startswith(prefix): continue
        key = name[len(prefix):]
        if key in expert_keys_in_sd: continue
        if key not in layer_sd: continue
        if arr.dtype == np.float32:
            layer_sd[key].copy_(torch.from_numpy(arr.copy()))
        else:
            f32 = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            layer_sd[key].copy_(torch.from_numpy(f32.copy()))

    # C) Move only what we need to GPU
    input_norm = layer.input_layernorm.to(DEVICE)
    post_norm = layer.post_attention_layernorm.to(DEVICE)
    layer.block_type = block_type  # dispatch for forward()

    if block_type == "full_attention":
        attn_module = layer.self_attn.to(DEVICE)
    else:
        attn_module = layer.linear_attn.to(DEVICE)

    lsd = layer.state_dict()
    router_w = lsd['mlp.gate.weight'].to(DEVICE)
    sgw = lsd['mlp.shared_expert_gate.weight'].to(DEVICE)
    sg  = lsd['mlp.shared_expert.gate_proj.weight'].to(DEVICE)
    su  = lsd['mlp.shared_expert.up_proj.weight'].to(DEVICE)
    sd_ = lsd['mlp.shared_expert.down_proj.weight'].to(DEVICE)

    pos_ids = torch.arange(0, hidden.shape[1], device=DEVICE).unsqueeze(0)
    pos_emb = rotary(hidden, pos_ids)

    # D) Forward
    with torch.no_grad():
        # Attention
        t3 = time.perf_counter()
        residual = hidden.clone()
        hn = input_norm(hidden)
        if block_type == "full_attention":
            a_out, _ = attn_module(hn, attention_mask=None,
                                    position_embeddings=pos_emb, position_ids=pos_ids)
        else:
            a_out = attn_module(hn, attention_mask=None)
        hidden = residual + a_out
        t4 = time.perf_counter()

        # MoE
        residual2 = hidden.clone()
        flat = post_norm(hidden).reshape(-1, 4096)
        moe_out = moe.forward(flat, router_w, lid, sgw, sg, su, sd_)
        hidden = residual2 + moe_out.reshape_as(residual2)
        t5 = time.perf_counter()

    times_ssd.append(t2 - t1)
    times_attn.append(t4 - t3)
    times_moe.append(t5 - t4)

    has_nan = torch.isnan(hidden).any().item()
    if has_nan: nan_layers.append(lid)

    block_label = "full" if block_type == "full_attention" else "lin"
    r = hidden.float().cpu().numpy()
    print(f"L{lid:02d} {block_label:4s} SSD={t2-t1:.2f}s Attn={t4-t3:.3f}s MoE={t5-t4:.3f}s "
          f"mean={r.mean():.4f} NaN={has_nan}")

    # Cleanup GPU
    del input_norm, post_norm, attn_module, router_w, sgw, sg, su, sd_, a_out, hn, flat, moe_out, residual, residual2
    torch.cuda.empty_cache()

# ── Summary ──
total = sum(times_ssd) + sum(times_attn) + sum(times_moe)
print(f"\n{'='*60}")
print(f"SSD: {sum(times_ssd):.0f}s | Attn: {sum(times_attn):.1f}s | MoE: {sum(times_moe):.1f}s")
print(f"Total: {total:.1f}s | NaN layers: {nan_layers if nan_layers else 'NONE'}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
status = "PASSED" if not nan_layers else f"FAILED at layers {nan_layers}"
print(f"Status: {status}")

streamer.close(); reader.close()
torch.cuda.empty_cache()
