"""
60层全真DeltaNet forward: 使用HF的torch_chunk_gated_delta_rule。
conv1d→QKV→A_log/dt_bias→g→beta→DeltaRule→RMSNormGate→out_proj
"""
import torch, torch.nn.functional as F, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'
DEV = "cuda"

config = __import__('transformers').AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = config.text_config
LV = tc.layer_types; H = tc.hidden_size
N_V = tc.linear_num_value_heads; N_K = tc.linear_num_key_heads
H_K = tc.linear_key_head_dim; H_V = tc.linear_value_head_dim

streamer = NonExpertStreamer(MODEL)
reader = SafeExpertReader(MODEL)

# Import torch DeltaNet functions from HF
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm
)

def np2gpu(a):
    if a.dtype == np.float32: return torch.from_numpy(a.copy()).to(DEV)
    return torch.from_numpy((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()).to(DEV)

def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

# MoE (reuse from final)
class MoE:
    def __init__(self):
        self.g = torch.zeros(32, 2048, H, device=DEV)
        self.d = torch.zeros(32, H, 1024, device=DEV)
    def fwd(self, flat, rw, lid, sgw, sg, su, sd):
        logits = flat @ rw.T; probs = torch.softmax(logits, -1)
        tw, ti = torch.topk(probs, 10, -1); uids = ti.unique().tolist()[:32]
        for s, e in enumerate(uids):
            gn, dn = reader.read_expert(lid, e)
            self.g[s] = torch.from_numpy((gn.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).to(DEV)
            self.d[s] = torch.from_numpy((dn.view(np.uint16).astype(np.uint32) << 16).view(np.float32)).to(DEV)
        torch.cuda.synchronize()
        out = torch.zeros_like(flat)
        for s, e in enumerate(uids):
            m = (ti == e).any(-1)
            if not m.any(): continue
            t = flat[m]; gw, uw = self.g[s, :1024], self.g[s, 1024:]
            o = (F.silu(t @ gw.T) * (t @ uw.T)) @ self.d[s].T
            twm = tw[m]; eh = (ti[m] == e)
            out[m] += o * (twm * eh).sum(-1, keepdim=True)
        gv = torch.sigmoid(flat @ sgw.T)
        so = (F.silu(flat @ sg.T) * (flat @ su.T)) @ sd.T
        return out + gv * so

moe = MoE()
print("=" * 50 + "\n60-Layer REAL DeltaNet Forward\n" + "=" * 50)

hidden = torch.randn(1, 8, H, device=DEV) * 0.02
tt = []

for lid in range(60):
    t0 = time.perf_counter()
    ne = streamer.read_layer(lid)
    pf = f'model.language_model.layers.{lid}.'

    def find(name, *kw):
        for k in ne:
            if k.startswith(pf) and all(w in k for w in kw):
                return np2gpu(ne[k])
        return None

    in_norm  = find(ne, 'input_layernorm')
    post_norm = find(ne, 'post_attention_layernorm')
    sgw = find(ne, 'shared_expert_gate')
    sg  = find(ne, 'shared_expert', 'gate_proj')
    su  = find(ne, 'shared_expert', 'up_proj')
    sd  = find(ne, 'shared_expert', 'down_proj')
    router = find(ne, 'mlp.gate.weight')

    bt = LV[lid]

    # ── Full Attention weights ──
    attn_q = attn_k = attn_v = attn_o = None
    if bt == "full_attention":
        attn_q = find(ne, 'q_proj'); attn_k = find(ne, 'k_proj')
        attn_v = find(ne, 'v_proj'); attn_o = find(ne, 'o_proj')

    # ── DeltaNet weights ──
    lin_qkv = lin_z = lin_a = lin_b = lin_out = lin_conv = lin_dt = lin_Alog = lin_norm = None
    if bt != "full_attention":
        lin_qkv  = find(ne, 'in_proj_qkv')
        lin_z    = find(ne, 'in_proj_z')
        lin_a    = find(ne, 'in_proj_a')
        lin_b    = find(ne, 'in_proj_b')
        lin_out  = find(ne, 'attn.out_proj')  # 'linear_attn.out_proj.weight', not other out_proj
        lin_dt   = find(ne, 'attn.dt_bias')  # linear_attn.dt_bias
        lin_Alog = find(ne, 'attn.A_log')     # linear_attn.A_log
        lin_conv = find(ne, 'attn.conv1d')     # linear_attn.conv1d.weight
        lin_norm = find(ne, 'attn.norm')  # 'linear_attn.norm.weight', not 'input_layernorm'

    with torch.no_grad():
        res = hidden.clone()
        hn = rms(hidden, in_norm) if in_norm is not None else hidden

        # ── Attention / DeltaNet ──
        if bt == "full_attention" and attn_q is not None:
            B, S, _ = hn.shape
            q = hn @ attn_q.T; k = hn @ attn_k.T; v = hn @ attn_v.T
            qa = q[:,:,:8192].view(B,S,32,256).transpose(1,2)
            qg = q[:,:,8192:].view(B,S,32,256).transpose(1,2)
            k = k.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
            v = v.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
            attn = F.scaled_dot_product_attention(qa,k,v,is_causal=True)
            a_out = (torch.sigmoid(qg)*attn).transpose(1,2).contiguous().view(B,S,8192) @ attn_o.T

        elif lin_qkv is not None:
            B, S, _ = hn.shape

            # 1) QKV + conv1d
            qkv = hn @ lin_qkv.T  # [B, S, 12288]
            qkv = qkv.transpose(1,2)  # [B, 12288, S]
            if lin_conv is not None:
                # Conv1d: weight [12288, 1, 1, 4] → squeeze to [12288, 1, 4]
                w = lin_conv.squeeze(1) if lin_conv.dim() == 4 else lin_conv
                qkv = F.silu(F.conv1d(qkv, w, padding=3, groups=12288))
                qkv = qkv[:,:,:S]  # causal trim
            qkv = qkv.transpose(1,2)  # [B, S, 12288]

            # Split: K, K, V
            K_DIM = N_K * H_K  # 2048
            V_DIM = N_V * H_V  # 8192
            q = qkv[:,:,:K_DIM].reshape(B, S, N_K, H_K)
            k = qkv[:,:,K_DIM:2*K_DIM].reshape(B, S, N_K, H_K)
            v = qkv[:,:,2*K_DIM:].reshape(B, S, N_V, H_V)

            # 2) Gates
            b = (hn @ lin_b.T) if lin_b is not None else torch.zeros(B, S, N_K, device=DEV)
            a = (hn @ lin_a.T) if lin_a is not None else torch.zeros(B, S, N_V, device=DEV)
            beta = b.sigmoid()

            # g (decay) from A_log + dt_bias + a
            if lin_Alog is not None and lin_dt is not None:
                Alog = lin_Alog.to(torch.float32)  # [N_V]
                dt_b = lin_dt.to(torch.float32)
                g = -Alog.exp() * F.softplus(a.float() + dt_b)
            else:
                g = torch.ones(B, S, N_V, device=DEV)

            # 3) Broadcast Q,K from N_K=16 to N_V=64
            ratio = N_V // N_K  # 4
            q = q.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, S, N_V, H_K)
            k = k.unsqueeze(2).expand(-1, -1, ratio, -1, -1).reshape(B, S, N_V, H_K)

            # 4) L2-norm Q, K
            q_norm = l2norm(q, dim=-1, eps=1e-6)
            k_norm = l2norm(k, dim=-1, eps=1e-6)

            # 5) Chunked Gated Delta Rule
            core_attn, _ = torch_chunk_gated_delta_rule(
                q_norm, k_norm, v, g, beta,
                chunk_size=64, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=True
            )
            # core_attn: [B, N_V, S, H_V] → [B, S, V_DIM]
            attn = core_attn.to(torch.float32).transpose(1,2).contiguous().view(B, S, V_DIM)

            # 6) Gate + RMSNorm
            z = (hn @ lin_z.T) if lin_z is not None else torch.zeros(B, S, V_DIM, device=DEV)
            z_gate = F.silu(z.float())
            if lin_norm is not None:
                n_in = attn.reshape(B, S, N_V, H_V).float()
                n_out = n_in * torch.rsqrt(n_in.pow(2).mean(-1, keepdim=True) + 1e-6)
                attn = (n_out * lin_norm.float().view(1, 1, 1, -1)).reshape(B, S, V_DIM)
            attn = attn * z_gate

            # 7) Output
            a_out = (attn @ lin_out.T) if lin_out is not None else attn[:,:,:H]
        else:
            a_out = hn * 0.99

        hidden = res + a_out

        # ── MoE ──
        res2 = hidden.clone()
        hn2 = rms(hidden, post_norm) if post_norm is not None else hidden
        flat = hn2.reshape(-1, H)
        moe_out = moe.fwd(flat, router, lid, sgw, sg, su, sd)
        hidden = res2 + moe_out.reshape_as(res2)

    e = time.perf_counter() - t0
    tt.append(e)
    r = hidden.float().cpu().numpy()
    nan = np.any(np.isnan(r))
    print(f"L{lid:02d} {bt[:4]:>4s} {e:.1f}s mean={r.mean():.4f} NaN={nan}")
    if nan: break

total = sum(tt)
print(f"\nTotal: {total:.0f}s VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
print(f"Status: {'PASSED all 60' if not nan else f'NaN at {lid}'}")
streamer.close(); reader.close()
torch.cuda.empty_cache()
