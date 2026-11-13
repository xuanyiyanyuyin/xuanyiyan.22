"""
KV Cache + Recurrent State: 5步decode, 逐token验证DeltaNet状态稳定。
"""
import torch, torch.nn.functional as F, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm
)

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'; DEV = "cuda"
config = __import__('transformers').AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = config.text_config; LV = tc.layer_types; H = tc.hidden_size
N_V=64; N_K=16; H_K=128; H_V=128

streamer = NonExpertStreamer(MODEL); reader = SafeExpertReader(MODEL)

def np2gpu(a):
    if a.dtype == np.float32: return torch.from_numpy(a.copy()).to(DEV)
    return torch.from_numpy((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()).to(DEV)
def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + 1e-6) * w

# ── KV Cache ──
class KVCache:
    def __init__(self, n_layers=60, max_seq=64):
        self.kv = {}  # lid -> (K, V) for full_attn layers
        self.conv = {}  # lid -> conv_state for DeltaNet layers
        self.recur = {}  # lid -> recurrent_state for DeltaNet layers
        self.seq_len = 0

    def get_kv(self, lid): return self.kv.get(lid)
    def get_conv(self, lid): return self.conv.get(lid)
    def get_recur(self, lid): return self.recur.get(lid)

    def update_kv(self, lid, new_k, new_v):
        if lid in self.kv:
            self.kv[lid] = (torch.cat([self.kv[lid][0], new_k], dim=2),
                            torch.cat([self.kv[lid][1], new_v], dim=2))
        else:
            self.kv[lid] = (new_k, new_v)

    def update_conv(self, lid, state):
        self.conv[lid] = state
    def update_recur(self, lid, state):
        self.recur[lid] = state

kv = KVCache()

# MoE (reuse)
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
            t = flat[m]; gw, uw = self.g[s,:1024], self.g[s,1024:]
            o = (F.silu(t @ gw.T) * (t @ uw.T)) @ self.d[s].T
            twm = tw[m]; eh = (ti[m] == e)
            out[m] += o * (twm * eh).sum(-1, keepdim=True)
        gv = torch.sigmoid(flat @ sgw.T)
        so = (F.silu(flat @ sg.T) * (flat @ su.T)) @ sd.T
        return out + gv * so

moe = MoE()
print("=" * 50 + "\nKV Cache + Recurrent State Test\n" + "=" * 50)

# ── Prefill (3 tokens) then decode (2 tokens) ──
hidden = torch.randn(1, 3, H, device=DEV) * 0.02

for step in range(5):
    seq_len = hidden.shape[1]
    is_prefill = step == 0

    print(f"\n{'Prefill' if is_prefill else 'Decode'}: step {step}, seq_len={seq_len}")
    t0 = time.perf_counter()

    for lid in range(60):
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

        attn_q = attn_k = attn_v = attn_o = None
        lin_qkv = lin_z = lin_a = lin_b = lin_out = lin_conv = lin_dt = lin_Alog = lin_norm = None

        if bt == "full_attention":
            attn_q = find(ne, 'q_proj'); attn_k = find(ne, 'k_proj')
            attn_v = find(ne, 'v_proj'); attn_o = find(ne, 'o_proj')
        else:
            lin_qkv  = find(ne, 'in_proj_qkv'); lin_z = find(ne, 'in_proj_z')
            lin_a    = find(ne, 'in_proj_a'); lin_b = find(ne, 'in_proj_b')
            lin_out  = find(ne, 'attn.out_proj')
            lin_dt   = find(ne, 'attn.dt_bias'); lin_Alog = find(ne, 'attn.A_log')
            lin_conv = find(ne, 'attn.conv1d'); lin_norm = find(ne, 'attn.norm')

        with torch.no_grad():
            res = hidden.clone()
            hn = rms(hidden, in_norm) if in_norm is not None else hidden
            B, S, _ = hn.shape

            # ── Attention ──
            if bt == "full_attention" and attn_q is not None:
                q = hn @ attn_q.T; k = hn @ attn_k.T; v = hn @ attn_v.T
                qa = q[:,:,:8192].view(B,S,32,256).transpose(1,2)
                qg = q[:,:,8192:].view(B,S,32,256).transpose(1,2)
                k = k.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
                v = v.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)

                # KV Cache
                if not is_prefill:
                    k_cache, v_cache = kv.get_kv(lid)
                    k = torch.cat([k_cache, k], dim=2)
                    v = torch.cat([v_cache, v], dim=2)
                kv.update_kv(lid, k, v)

                attn = F.scaled_dot_product_attention(qa, k, v, is_causal=is_prefill)
                a_out = (torch.sigmoid(qg)*attn).transpose(1,2).contiguous().view(B,S,8192) @ attn_o.T

            elif lin_qkv is not None:
                K_DIM = N_K * H_K; V_DIM = N_V * H_V

                qkv = hn @ lin_qkv.T
                qkv = qkv.transpose(1,2)

                # Conv1d with state
                if lin_conv is not None:
                    w = lin_conv.squeeze(1) if lin_conv.dim() == 4 else lin_conv
                    if not is_prefill:
                        conv_state = kv.get_conv(lid)
                        if conv_state is not None:
                            qkv = torch.cat([conv_state, qkv], dim=-1)
                    qkv = F.silu(F.conv1d(qkv, w, padding=3, groups=12288))[:,:,:S]
                    kv.update_conv(lid, qkv[:,:,-3:].clone())  # last 3 for next step

                qkv = qkv.transpose(1,2)
                q = qkv[:,:,:K_DIM].reshape(B,S,N_K,H_K)
                k = qkv[:,:,K_DIM:2*K_DIM].reshape(B,S,N_K,H_K)
                v = qkv[:,:,2*K_DIM:].reshape(B,S,N_V,H_V)

                b = (hn @ lin_b.T) if lin_b is not None else torch.zeros(B,S,N_K,device=DEV)
                a = (hn @ lin_a.T) if lin_a is not None else torch.zeros(B,S,N_V,device=DEV)
                beta = b.sigmoid()
                Alog = lin_Alog.float() if lin_Alog is not None else torch.zeros(N_V,device=DEV)
                dt_b = lin_dt.float() if lin_dt is not None else torch.zeros(N_V,device=DEV)
                g = -Alog.exp() * F.softplus(a.float() + dt_b)

                ratio = N_V // N_K
                q = q.unsqueeze(2).expand(-1,-1,ratio,-1,-1).reshape(B,S,N_V,H_K)
                k = k.unsqueeze(2).expand(-1,-1,ratio,-1,-1).reshape(B,S,N_V,H_K)

                q_n, k_n = l2norm(q, -1, 1e-6), l2norm(k, -1, 1e-6)

                # Recurrent state
                init_state = None if is_prefill else kv.get_recur(lid)
                core_attn, last_state = torch_chunk_gated_delta_rule(
                    q_n, k_n, v, g, beta, chunk_size=64,
                    initial_state=init_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True
                )
                kv.update_recur(lid, last_state)

                attn = core_attn.float().transpose(1,2).contiguous().view(B,S,V_DIM)
                z = (hn @ lin_z.T) if lin_z is not None else torch.zeros(B,S,V_DIM,device=DEV)
                if lin_norm is not None:
                    n_in = attn.reshape(B,S,N_V,H_V).float()
                    attn = (n_in * torch.rsqrt(n_in.pow(2).mean(-1,keepdim=True)+1e-6) * lin_norm.float().view(1,1,1,-1)).reshape(B,S,V_DIM)
                attn = attn * F.silu(z.float())
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
    r = hidden.float().cpu().numpy()
    r_last = r[0, -1, :]
    nan = np.any(np.isnan(r))
    print(f"  Time={e:.1f}s mean={r.mean():.4f} last_token_std={r_last.std():.6f} NaN={nan}")
    if nan: break

    # Next step: feed last token only
    if step < 4:
        hidden = hidden[:, -1:, :]  # keep only last token for decode

print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
print(f"Status: {'PASSED' if not nan else 'FAILED'}")
streamer.close(); reader.close()
