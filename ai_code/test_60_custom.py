"""
纯 PyTorch 60层 forward: 不依赖 HF Qwen3_5MoeDecoderLayer。
用 streamer 加载权重, 直接用 torch 操作计算。
"""
import torch, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from layer_utils import load_layer_types

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
DEVICE = "cuda"

from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = config.text_config
LV = tc.layer_types
HIDDEN = tc.hidden_size  # 4096
N_HEADS = tc.num_attention_heads  # 32
N_KV = tc.num_key_value_heads  # 2
HEAD_DIM = tc.head_dim  # 256
INTER = tc.moe_intermediate_size  # 1024

streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)


def np_to_gpu(arr):
    if arr.dtype == np.float32:
        return torch.from_numpy(arr.copy()).to(DEVICE)
    return torch.from_numpy((arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()).to(DEVICE)


# ── RMSNorm ──
def rms_norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


# ── Full Attention ──
def full_attn_forward(h, lw):
    q = h @ lw['q_proj'].T  # [B,S,16384]
    k = h @ lw['k_proj'].T  # [B,S,512]
    v = h @ lw['v_proj'].T  # [B,S,512]
    # Q split: 32 attn + 32 gate, each head_dim=256
    B, S, _ = q.shape
    q_attn = q[:, :, :8192].view(B, S, 32, 256).transpose(1, 2)  # 32 heads
    q_gate = q[:, :, 8192:].view(B, S, 32, 256).transpose(1, 2)
    k = k.view(B, S, 2, 256).transpose(1, 2).repeat_interleave(16, 1)
    v = v.view(B, S, 2, 256).transpose(1, 2).repeat_interleave(16, 1)
    attn = torch.nn.functional.scaled_dot_product_attention(q_attn, k, v, is_causal=True)
    attn = torch.sigmoid(q_gate) * attn
    attn = attn.transpose(1, 2).contiguous().view(B, S, 8192)
    return attn @ lw['o_proj'].T


# ── Linear Attention (simplified: RMSNorm + decay, passing shape only) ──
def lin_attn_forward(h, lw):
    # Complete DeltaNet requires SSM scan + convolution.
    # For pipeline test: stable pass-through preserving shape.
    return h * 0.99


# ── Scattered MoE ──
class ScatteredMoE:
    def __init__(self, max_unique=32):
        self.gu = torch.zeros(max_unique, 2048, HIDDEN, device=DEVICE)
        self.dn = torch.zeros(max_unique, HIDDEN, 1024, device=DEVICE)

    def forward(self, h_flat, router_w, lid):
        logits = h_flat @ router_w.T
        probs = torch.softmax(logits, -1)
        tw, ti = torch.topk(probs, 10, -1)
        uids = ti.unique().tolist()[:32]

        for s, eid in enumerate(uids):
            gn, dn_np = reader.read_expert(lid, eid)
            self.gu[s] = torch.from_numpy((gn.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).to(DEVICE)
            self.dn[s] = torch.from_numpy((dn_np.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).to(DEVICE)
        torch.cuda.synchronize()

        out = torch.zeros_like(h_flat)
        for s, eid in enumerate(uids):
            m = (ti == eid).any(-1)
            if not m.any(): continue
            t = h_flat[m]
            gw, uw = self.gu[s, :1024], self.gu[s, 1024:]
            gate = torch.nn.functional.silu(t @ gw.T)
            up = t @ uw.T
            o = (gate * up) @ self.dn[s].T
            tw_m = tw[m]; eh = (ti[m] == eid)
            o = o * (tw_m * eh).sum(-1, keepdim=True)
            out[m] += o

        # Shared expert
        sgw = lw_shared['sgw']; sg = lw_shared['sg']; su = lw_shared['su']; sd = lw_shared['sd']
        gv = torch.sigmoid(h_flat @ sgw.T)
        so = (torch.nn.functional.silu(h_flat @ sg.T) * (h_flat @ su.T)) @ sd.T
        return out + gv * so


# ── Main ──
print("=" * 60)
print("60-Layer Pure PyTorch Forward")
print("=" * 60)

hidden = torch.randn(1, 8, HIDDEN, device=DEVICE) * 0.02
moe = ScatteredMoE()
times = []

for lid in range(60):
    t0 = time.perf_counter()

    # Load non-expert weights
    ne = streamer.read_layer(lid)
    prefix = f'model.language_model.layers.{lid}.'
    lw = {}
    for name, arr in ne.items():
        if not name.startswith(prefix): continue
        key = name[len(prefix):]
        if 'experts' in key and 'shared' not in key: continue
        # Simplify key: model.language_model.layers.{L}.self_attn.q_proj.weight → q_proj.weight
        # Or: input_layernorm.weight → input_layernorm.weight
        parts = key.split('.')
        if len(parts) >= 2:
            short = parts[-2]  # 'q_proj', 'k_proj', 'o_proj', 'input_layernorm', etc
        else:
            short = key
        lw[short] = np_to_gpu(arr)

    # Extract shared expert + router weights from lw
    sgw = lw.pop('weight', None)  # shared_expert_gate.weight
    sg  = lw.pop('weight', None)  # might collide — need better key
    # Better: use the full key from streamer
    sgw = None; sg = None; su = None; sdo = None
    router_w = None
    for name, arr in ne.items():
        if not name.startswith(prefix): continue
        short = name.split('.')[-2]
        if 'shared_expert_gate' in name: sgw = np_to_gpu(arr)
        elif 'shared_expert.gate_proj' in name: sg = np_to_gpu(arr)
        elif 'shared_expert.up_proj' in name: su = np_to_gpu(arr)
        elif 'shared_expert.down_proj' in name: sdo = np_to_gpu(arr)
        elif 'mlp.gate.weight' in name: router_w = np_to_gpu(arr)
    lw_shared = {'sgw': sgw, 'sg': sg, 'su': su, 'sd': sdo}
    # router_w already set above

    # Forward
    with torch.no_grad():
        # Attention block
        residual = hidden.clone()
        inw = lw.get('input_layernorm.weight')
        if inw is not None:
            hn = rms_norm(hidden, inw)
        else:
            hn = hidden

        bt = LV[lid]
        if bt == "full_attention":
            a_out = full_attn_forward(hn, lw)
        else:
            a_out = lin_attn_forward(hn, lw)
        hidden = residual + a_out

        # MoE block
        residual2 = hidden.clone()
        pnw = lw.get('post_attention_layernorm.weight')
        if pnw is not None:
            hn2 = rms_norm(hidden, pnw)
        else:
            hn2 = hidden

        flat = hn2.reshape(-1, HIDDEN)
        moe_out = moe.forward(flat, router_w, lid)
        hidden = residual2 + moe_out.reshape_as(residual2)

    elapsed = time.perf_counter() - t0
    times.append(elapsed)

    r = hidden.float().cpu().numpy()
    nan = np.any(np.isnan(r))
    bt_label = "full" if bt == "full_attention" else "lin"
    print(f"L{lid:02d} {bt_label:4s} {elapsed:.2f}s mean={r.mean():.4f} NaN={nan}")
    if nan: break

    # Free GPU
    del lw, router_w, sgw, sg, su, sd_, a_out, hn, flat, moe_out, residual, residual2
    torch.cuda.empty_cache()

# Summary
total = sum(times)
print(f"\nTotal: {total:.1f}s | NaN layers: {lid if nan else 'NONE'}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
if not nan: print("STATUS: ALL 60 LAYERS PASSED")

streamer.close(); reader.close()
torch.cuda.empty_cache()
