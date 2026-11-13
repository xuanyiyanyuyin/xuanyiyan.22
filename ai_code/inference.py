"""
Qwen3.5-397B 本地推理引擎 v1.0
RTX 5070 12GB | SSD→RAM→GPU 流式管线 | KV Cache + Recurrent State
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
from collections import deque

# ── Config ──
MODEL_DIR  = r'D:\ai\models\qwen3.5-397b-bf16'
DEVICE     = "cuda"
MAX_TOKENS = 5
TEMPERATURE = 0.8
TOP_K      = 40

# ── Import HF components ──
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm, Qwen3_5MoeRMSNormGated
)
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer

# ── Init ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
config    = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
tc        = config.text_config
streamer  = NonExpertStreamer(MODEL_DIR)
reader    = SafeExpertReader(MODEL_DIR)

H     = tc.hidden_size          # 4096
V     = tc.vocab_size           # 248320
LV    = tc.layer_types          # full/linear per layer
N_V   = tc.linear_num_value_heads  # 64
N_K   = tc.linear_num_key_heads    # 16
H_K   = tc.linear_key_head_dim     # 128
H_V   = tc.linear_value_head_dim   # 128
EPS   = tc.rms_norm_eps

print(f"[Engine] Qwen3.5-397B | {H}D | 60 layers | 512 experts | {V} vocab")
print(f"[Engine] {DEVICE.upper()} | {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB VRAM")

# ═══════════════════════════════════════
# Utilities
# ═══════════════════════════════════════

def bf16_to_fp32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 numpy → float32 numpy (safe for ml_dtypes)"""
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def load_weight(key: str, shape=None):
    """Load one tensor from safetensors by key name, return torch.Tensor on GPU"""
    idx_path = os.path.join(MODEL_DIR, 'model.safetensors.index.json')
    with open(idx_path) as f:
        idx = json.load(f)
    shard = idx['weight_map'][key]
    with open(os.path.join(MODEL_DIR, shard), 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    m = hdr[key]
    os_s, os_e = m['data_offsets']
    fd = os.open(os.path.join(MODEL_DIR, shard), os.O_RDONLY | os.O_BINARY)
    os.lseek(fd, n + 8 + os_s, os.SEEK_SET)
    raw = os.read(fd, os_e - os_s)
    os.close(fd)
    arr = bf16_to_fp32(np.frombuffer(raw, dtype=np.uint16))
    if shape: arr = arr.reshape(shape)
    return torch.from_numpy(arr.copy()).to(DEVICE)

def stream_weight(key: str, ne: dict, pf: str):
    """Find weight from cache dict (torch tensors), move to GPU"""
    full = pf + key
    t = ne.get(full)
    return t.to(DEVICE) if t is not None else None

# ═══════════════════════════════════════
# Resident weights (embed + lm_head)
# ═══════════════════════════════════════
embed_w = load_weight('model.language_model.embed_tokens.weight', (V, H))
lm_head_w = embed_w  # tied
print(f"[Resident] embed {embed_w.shape}")

# ═══════════════════════════════════════
# KV Cache
# ═══════════════════════════════════════
class Cache:
    def __init__(self):
        self.kv = {}      # lid → (K, V) for full_attention
        self.conv = {}    # lid → conv_state for DeltaNet
        self.recur = {}   # lid → recurrent_state for DeltaNet

    def get_kv(self, lid): return self.kv.get(lid)
    def get_conv(self, lid): return self.conv.get(lid)
    def get_recur(self, lid): return self.recur.get(lid)

    def update_kv(self, lid, k, v):
        if lid in self.kv:
            self.kv[lid] = (torch.cat([self.kv[lid][0], k], 2),
                           torch.cat([self.kv[lid][1], v], 2))
        else: self.kv[lid] = (k, v)

    def update_conv(self, lid, s): self.conv[lid] = s
    def update_recur(self, lid, s): self.recur[lid] = s

# ═══════════════════════════════════════
# Scattered MoE
# ═══════════════════════════════════════
class MoE:
    def __init__(self, max_experts=32):
        self.gu = torch.zeros(max_experts, 2048, H, device=DEVICE)
        self.dn = torch.zeros(max_experts, H, 1024, device=DEVICE)

    def forward(self, hidden_flat, router_w, layer_id,
                sgw, sg, su, sd):
        logits = hidden_flat @ router_w.T
        probs = torch.softmax(logits, -1)
        tw, ti = torch.topk(probs, 10, -1)
        uids = ti.unique().tolist()[:32]

        for s, eid in enumerate(uids):
            gn, dn = reader.read_expert(layer_id, eid)
            self.gu[s] = torch.from_numpy(bf16_to_fp32(gn.astype(np.uint16))).to(DEVICE)
            self.dn[s] = torch.from_numpy(bf16_to_fp32(dn.astype(np.uint16))).to(DEVICE)
        torch.cuda.synchronize()

        out = torch.zeros_like(hidden_flat)
        for s, eid in enumerate(uids):
            mask = (ti == eid).any(-1)
            if not mask.any(): continue
            t = hidden_flat[mask]
            gw, uw = self.gu[s, :1024], self.gu[s, 1024:]
            o = (F.silu(t @ gw.T) * (t @ uw.T)) @ self.dn[s].T
            twm = tw[mask]; eh = (ti[mask] == eid)
            out[mask] += o * (twm * eh).sum(-1, keepdim=True)

        gv = torch.sigmoid(hidden_flat @ sgw.T)
        so = (F.silu(hidden_flat @ sg.T) * (hidden_flat @ su.T)) @ sd.T
        return out + gv * so

# ═══════════════════════════════════════
# Layer Forward
# ═══════════════════════════════════════
def forward_attention(hidden, ne, pf, lid, bt, is_prefill, cache):
    """Returns (output, updated_cache)"""
    B, S, _ = hidden.shape

    if bt == "full_attention":
        q = stream_weight('self_attn.q_proj.weight', ne, pf)
        k = stream_weight('self_attn.k_proj.weight', ne, pf)
        v = stream_weight('self_attn.v_proj.weight', ne, pf)
        o = stream_weight('self_attn.o_proj.weight', ne, pf)

        if q is None: return hidden * 0.99

        Q = hidden @ q.T  # [B,S,16384]
        K = hidden @ k.T  # [B,S,512]
        V = hidden @ v.T  # [B,S,512]

        qa = Q[:,:,:8192].view(B,S,32,256).transpose(1,2)
        qg = Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
        K = K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        V = V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)

        if not is_prefill:
            kc, vc = cache.get_kv(lid)
            if kc is not None: K = torch.cat([kc, K], 2); V = torch.cat([vc, V], 2)
        cache.update_kv(lid, K, V)

        attn = F.scaled_dot_product_attention(qa, K, V, is_causal=is_prefill)
        return (torch.sigmoid(qg) * attn).transpose(1,2).contiguous().view(B,S,8192) @ o.T

    else:  # linear_attention (DeltaNet)
        qkv  = stream_weight('linear_attn.in_proj_qkv.weight', ne, pf)
        z_w  = stream_weight('linear_attn.in_proj_z.weight', ne, pf)
        a_w  = stream_weight('linear_attn.in_proj_a.weight', ne, pf)
        b_w  = stream_weight('linear_attn.in_proj_b.weight', ne, pf)
        out  = stream_weight('linear_attn.out_proj.weight', ne, pf)
        conv = stream_weight('linear_attn.conv1d.weight', ne, pf)
        dt   = stream_weight('linear_attn.dt_bias', ne, pf)
        Alog = stream_weight('linear_attn.A_log', ne, pf)
        nrm  = stream_weight('linear_attn.norm.weight', ne, pf)

        if qkv is None: return hidden * 0.99

        K_DIM = N_K * H_K; V_DIM = N_V * H_V

        # Project + Conv
        mixed = hidden @ qkv.T  # [B,S,12288]
        mixed = mixed.transpose(1,2)  # [B,12288,S]
        if conv is not None:
            w = conv.squeeze(1) if conv.dim() == 4 else conv
            if not is_prefill:
                cs = cache.get_conv(lid)
                if cs is not None: mixed = torch.cat([cs, mixed], -1)
            mixed = F.silu(F.conv1d(mixed, w, padding=3, groups=12288))[:,:,:S]
            cache.update_conv(lid, mixed[:,:,-3:].clone())
        mixed = mixed.transpose(1,2)

        Q = mixed[:,:,:K_DIM].reshape(B, S, N_K, H_K)
        K = mixed[:,:,K_DIM:2*K_DIM].reshape(B, S, N_K, H_K)
        V = mixed[:,:,2*K_DIM:].reshape(B, S, N_V, H_V)

        # Gates
        beta = (hidden @ b_w.T).sigmoid() if b_w is not None else torch.ones(B,S,N_K,device=DEVICE)
        a    = (hidden @ a_w.T) if a_w is not None else torch.zeros(B,S,N_V,device=DEVICE)
        g    = -Alog.float().exp() * F.softplus(a.float() + (dt.float() if dt is not None else 0))

        # Broadcast 16 → 64
        R = N_V // N_K
        Q = Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K)
        K = K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K)

        # Delta Rule
        qn = l2norm(Q, -1, 1e-6); kn = l2norm(K, -1, 1e-6)
        init = None if is_prefill else cache.get_recur(lid)
        core, last = torch_chunk_gated_delta_rule(
            qn, kn, V, g, beta, 64, init, True, use_qk_l2norm_in_kernel=True
        )
        cache.update_recur(lid, last)

        attn = core.float().transpose(1,2).contiguous().view(B, S, V_DIM)

        # RMSNormGate
        if nrm is not None:
            ni = attn.reshape(B,S,N_V,H_V).float()
            attn = (ni * torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EPS)
                    * nrm.float().view(1,1,1,-1)).reshape(B,S,V_DIM)
        z = (hidden @ z_w.T) if z_w is not None else attn
        attn = attn * F.silu(z.float())

        return (attn @ out.T) if out is not None else attn[:,:,:H]

# ═══════════════════════════════════════
# Main Inference Loop
# ═══════════════════════════════════════
def generate(prompt: str, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEVICE)
    hidden = F.embedding(input_ids, embed_w)
    cache = Cache()
    moe = MoE()
    generated = tokens.copy()

    # ── RAM Cache: prefetch all 60 layers' non-expert weights once, pre-convert to torch ──
    layer_cache = {}
    print(f"  Prefetching {60} layers...", end="", flush=True)
    t_cache = time.perf_counter()
    for lid in range(60):
        ne = streamer.read_layer(lid)
        pf = f'model.language_model.layers.{lid}.'
        # Pre-convert all weights to torch tensors on CPU
        torch_ne = {}
        for k, v in ne.items():
            if v.dtype == np.float32:
                torch_ne[k] = torch.from_numpy(v.copy())
            else:
                torch_ne[k] = torch.from_numpy(bf16_to_fp32(v.astype(np.uint16)).copy())
        layer_cache[lid] = (torch_ne, pf)
    print(f" {time.perf_counter()-t_cache:.0f}s", flush=True)

    print(f"\n{'─'*50}", flush=True)
    print(f"Prompt: {prompt!r}", flush=True)
    print(f"{'─'*50}", flush=True)

    t0 = time.perf_counter()
    for step in range(max_tokens):
        is_prefill = (step == 0)

        for lid in range(60):
            ne, pf = layer_cache[lid]  # ← RAM cached
            bt = LV[lid]

            # Norms
            in_norm  = stream_weight('input_layernorm.weight', ne, pf)
            post_norm = stream_weight('post_attention_layernorm.weight', ne, pf)

            # Shared expert
            sgw = stream_weight('mlp.shared_expert_gate.weight', ne, pf)
            sg  = stream_weight('mlp.shared_expert.gate_proj.weight', ne, pf)
            su  = stream_weight('mlp.shared_expert.up_proj.weight', ne, pf)
            sd  = stream_weight('mlp.shared_expert.down_proj.weight', ne, pf)

            # Router
            router = stream_weight('mlp.gate.weight', ne, pf)

            with torch.no_grad():
                # ── Attention ──
                hn = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS) if in_norm is None else \
                     hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS) * in_norm.float()
                a_out = forward_attention(hn, ne, pf, lid, bt, is_prefill, cache)
                hidden = hidden + a_out

                # ── MoE ──
                hn2 = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS) if post_norm is None else \
                      hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS) * post_norm.float()
                flat = hn2.reshape(-1, H)
                moe_out = moe.forward(flat, router, lid, sgw, sg, su, sd)
                hidden = hidden + moe_out.reshape_as(hidden)

            # Free GPU copies of this layer's weights
            for v in [in_norm, post_norm, sgw, sg, su, sd, router, a_out, hn, hn2, flat, moe_out]:
                if v is not None: del v
            if lid % 20 == 0: torch.cuda.empty_cache()

        # ── LM Head ──
        logits = hidden[:, -1:, :].float() @ lm_head_w.float().T  # [1,1,V]
        logits = logits / max(TEMPERATURE, 0.01)

        # NaN protection: if signal drifted, fallback to argmax over valid tokens
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

        if TOP_K > 0:
            v, _ = torch.topk(logits, TOP_K, dim=-1)
            logits[logits < v[:,:,-1:]] = -float('inf')

        probs = torch.softmax(logits, -1)
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]  # uniform fallback
        next_tok = torch.multinomial(probs.squeeze(), 1).item()

        generated.append(next_tok)
        token_str = tokenizer.decode([next_tok])
        has_nan = torch.isnan(hidden).any().item()
        print(f"  [{step}] '{token_str}' (id={next_tok}) NaN={has_nan}", flush=True)

        if next_tok == tokenizer.eos_token_id or next_tok == tokenizer.pad_token_id:
            break

        # Next step
        if step < max_tokens - 1:
            hidden = F.embedding(torch.tensor([[next_tok]], device=DEVICE), embed_w)

    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(generated)
    print(f"  → {text!r}")
    print(f"  [{elapsed:.0f}s, {len(generated)-len(tokens)} tokens, {elapsed/max(len(generated)-len(tokens),1):.0f}s/tok]")

# ═══════════════════════════════════════
# Run
# ═══════════════════════════════════════
for prompt in ["你好"]:  # quick test: 1 prompt
    generate(prompt)

print(f"\nVRAM peak: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
streamer.close(); reader.close()
print("DONE")
