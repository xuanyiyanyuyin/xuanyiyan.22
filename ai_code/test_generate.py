"""
真实文本生成: Tokenizer + 60层 forward + KV Cache + 自回归decode。
"""
import torch, torch.nn.functional as F, time, sys, numpy as np
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm
)

MODEL = r'D:\ai\models\qwen3.5-397b-bf16'; DEV = "cuda"
config = __import__('transformers').AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc = config.text_config; LV = tc.layer_types; H = tc.hidden_size
N_V=64; N_K=16; H_K=128; H_V=128; VOCAB = tc.vocab_size

streamer = NonExpertStreamer(MODEL); reader = SafeExpertReader(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print(f"[Tokenizer] vocab={VOCAB} eos={tokenizer.eos_token_id}")

def np2gpu(a):
    if a.dtype == np.float32: return torch.from_numpy(a.copy()).to(DEV)
    return torch.from_numpy((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32).copy()).to(DEV)
def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + 1e-6) * w

# ── Embed & LM Head (resident) ──
import json, os, struct as st
with open(os.path.join(MODEL, 'model.safetensors.index.json')) as f:
    idx = json.load(f)
def load_one(key, shape=None):
    shard = idx['weight_map'][key]
    with open(os.path.join(MODEL, shard), 'rb') as f:
        n = st.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
    m = hdr[key]; os_s, os_e = m['data_offsets']
    fd = os.open(os.path.join(MODEL, shard), os.O_RDONLY | os.O_BINARY)
    os.lseek(fd, n+8+os_s, os.SEEK_SET); raw = os.read(fd, os_e-os_s); os.close(fd)
    t = np2gpu(np.frombuffer(raw, dtype=np.uint16))
    if shape: t = t.reshape(shape)
    return t

embed_w = load_one('model.language_model.embed_tokens.weight', (VOCAB, H))
lm_head_w = embed_w  # tied weights
print(f"[Embed] loaded {embed_w.shape}")

# KV Cache + MoE
class KVCache:
    def __init__(self):
        self.kv={}; self.conv={}; self.recur={}
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

kv = KVCache(); moe = MoE()

# ── Tokenize ──
for prompt in ["你好", "1+1=", "今天天气", "Python"]:
    tokens = tokenizer.encode(prompt)
    print(f"\n{'='*50}")
    print(f"Prompt: {prompt!r} → tokens={tokens}")
    print(f"{'='*50}")

    input_ids = torch.tensor([tokens], device=DEV)
    hidden = F.embedding(input_ids, embed_w)  # [1, seq, 4096]
    generated = tokens.copy()
    kv = KVCache()

    t0 = time.perf_counter()
    for step in range(5):  # generate 5 tokens
        is_prefill = step == 0
        S = hidden.shape[1]

        for lid in range(60):
            ne = streamer.read_layer(lid)
            pf = f'model.language_model.layers.{lid}.'
            def find(name, *kw):
                for k in ne:
                    if k.startswith(pf) and all(w in k for w in kw):
                        return np2gpu(ne[k])
                return None

            in_norm=find(ne,'input_layernorm'); post_norm=find(ne,'post_attention_layernorm')
            sgw=find(ne,'shared_expert_gate'); sg=find(ne,'shared_expert','gate_proj')
            su=find(ne,'shared_expert','up_proj'); sd=find(ne,'shared_expert','down_proj')
            router=find(ne,'mlp.gate.weight'); bt=LV[lid]

            attn_q=attn_k=attn_v=attn_o=None
            lin_qkv=lin_z=lin_a=lin_b=lin_out=lin_conv=lin_dt=lin_Alog=lin_norm=None
            if bt=="full_attention":
                attn_q=find(ne,'q_proj'); attn_k=find(ne,'k_proj')
                attn_v=find(ne,'v_proj'); attn_o=find(ne,'o_proj')
            else:
                lin_qkv=find(ne,'in_proj_qkv'); lin_z=find(ne,'in_proj_z')
                lin_a=find(ne,'in_proj_a'); lin_b=find(ne,'in_proj_b')
                lin_out=find(ne,'attn.out_proj')
                lin_dt=find(ne,'attn.dt_bias'); lin_Alog=find(ne,'attn.A_log')
                lin_conv=find(ne,'attn.conv1d'); lin_norm=find(ne,'attn.norm')

            with torch.no_grad():
                res=hidden.clone()
                hn=rms(hidden,in_norm) if in_norm is not None else hidden
                B,SL=hn.shape[:2]
                if bt=="full_attention" and attn_q is not None:
                    q=hn@attn_q.T; k=hn@attn_k.T; v=hn@attn_v.T
                    qa=q[:,:,:8192].view(B,SL,32,256).transpose(1,2)
                    qg=q[:,:,8192:].view(B,SL,32,256).transpose(1,2)
                    k=k.view(B,SL,2,256).transpose(1,2).repeat_interleave(16,1)
                    v=v.view(B,SL,2,256).transpose(1,2).repeat_interleave(16,1)
                    if not is_prefill:
                        kc,vc=kv.get_kv(lid); k=torch.cat([kc,k],2); v=torch.cat([vc,v],2)
                    kv.update_kv(lid,k,v)
                    attn=F.scaled_dot_product_attention(qa,k,v,is_causal=is_prefill)
                    a_out=(torch.sigmoid(qg)*attn).transpose(1,2).contiguous().view(B,SL,8192)@attn_o.T
                elif lin_qkv is not None:
                    K_DIM=N_K*H_K; V_DIM=N_V*H_V
                    qkv=hn@lin_qkv.T; qkv=qkv.transpose(1,2)
                    if lin_conv is not None:
                        w=lin_conv.squeeze(1) if lin_conv.dim()==4 else lin_conv
                        if not is_prefill:
                            cs=kv.get_conv(lid); qkv=torch.cat([cs,qkv],-1) if cs is not None else qkv
                        qkv=F.silu(F.conv1d(qkv,w,padding=3,groups=12288))[:,:,:SL]
                        kv.update_conv(lid,qkv[:,:,-3:].clone())
                    qkv=qkv.transpose(1,2)
                    q=qkv[:,:,:K_DIM].reshape(B,SL,N_K,H_K)
                    k=qkv[:,:,K_DIM:2*K_DIM].reshape(B,SL,N_K,H_K)
                    v=qkv[:,:,2*K_DIM:].reshape(B,SL,N_V,H_V)
                    b=(hn@lin_b.T) if lin_b is not None else torch.zeros(B,SL,N_K,device=DEV)
                    a=(hn@lin_a.T) if lin_a is not None else torch.zeros(B,SL,N_V,device=DEV)
                    beta=b.sigmoid()
                    Alog=lin_Alog.float() if lin_Alog is not None else torch.zeros(N_V,device=DEV)
                    dt_b=lin_dt.float() if lin_dt is not None else torch.zeros(N_V,device=DEV)
                    g=-Alog.exp()*F.softplus(a.float()+dt_b)
                    r=N_V//N_K
                    q=q.unsqueeze(2).expand(-1,-1,r,-1,-1).reshape(B,SL,N_V,H_K)
                    k=k.unsqueeze(2).expand(-1,-1,r,-1,-1).reshape(B,SL,N_V,H_K)
                    q_n,k_n=l2norm(q,-1,1e-6),l2norm(k,-1,1e-6)
                    init_s=None if is_prefill else kv.get_recur(lid)
                    ca,ls=torch_chunk_gated_delta_rule(q_n,k_n,v,g,beta,64,init_s,True,True)
                    kv.update_recur(lid,ls)
                    attn=ca.float().transpose(1,2).contiguous().view(B,SL,V_DIM)
                    z=(hn@lin_z.T) if lin_z is not None else torch.zeros(B,SL,V_DIM,device=DEV)
                    if lin_norm is not None:
                        ni=attn.reshape(B,SL,N_V,H_V).float()
                        attn=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+1e-6)*lin_norm.float().view(1,1,1,-1)).reshape(B,SL,V_DIM)
                    attn=attn*F.silu(z.float())
                    a_out=(attn@lin_out.T) if lin_out is not None else attn[:,:,:H]
                else:
                    a_out=hn*0.99
                hidden=res+a_out
                res2=hidden.clone()
                hn2=rms(hidden,post_norm) if post_norm is not None else hidden
                flat=hn2.reshape(-1,H)
                moe_out=moe.fwd(flat,router,lid,sgw,sg,su,sd)
                hidden=res2+moe_out.reshape_as(res2)

        # ── LM Head ──
        logits = (hidden[:,-1:,:].float() @ lm_head_w.float().T)  # [1,1,vocab]
        next_tok = logits.argmax(-1).item()
        generated.append(next_tok)
        print(f"  step {step}: '{tokenizer.decode([next_tok])}' (id={next_tok})")

        if next_tok == tokenizer.eos_token_id: break

        # Next: embed new token
        next_emb = F.embedding(torch.tensor([[next_tok]], device=DEV), embed_w)
        hidden = next_emb

    t = time.perf_counter() - t0
    text = tokenizer.decode(generated)
    print(f"  → {text!r} ({t:.0f}s)")

print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
streamer.close(); reader.close()
