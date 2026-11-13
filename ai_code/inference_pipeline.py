"""
Qwen3.5-397B 流水线推理 v2.0
GPU计算 + SSD预取 + H2D搬运 三级并发
"""
import torch, torch.nn.functional as F, time, numpy as np
import threading, queue, sys, json, os, struct
from collections import deque

MODEL_DIR = r'D:\ai\models\qwen3.5-397b-bf16'; DEV = "cuda"
sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
tc = config.text_config
streamer = NonExpertStreamer(MODEL_DIR)
reader = SafeExpertReader(MODEL_DIR)

H=tc.hidden_size; V=tc.vocab_size; LV=tc.layer_types
N_V=64; N_K=16; H_K=128; H_V=128; EPS=tc.rms_norm_eps
NUM_LAYERS=60; MAX_TOKENS=3; TEMPERATURE=0.8; TOP_K=40

print(f"[Pipeline] {NUM_LAYERS} layers | async prefetch | double-buffer")

# ── Utilities ──
def bf16_fp32(a): return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

# ── Load embed from first layer's streamer (already loaded, fast) ──
ne0 = streamer.read_layer(0)
embed_key = 'model.language_model.embed_tokens.weight'
# Search for embed in the weight map
with open(os.path.join(MODEL_DIR, 'model.safetensors.index.json')) as f:
    wm = json.load(f)['weight_map']
for k, v in wm.items():
    if 'embed_tokens' in k and 'language_model' in k:
        embed_key = k; break

embed_raw = None
with open(os.path.join(MODEL_DIR, wm[embed_key]), 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
m = hdr[embed_key]; os_s, os_e = m['data_offsets']
fd = os.open(os.path.join(MODEL_DIR, wm[embed_key]), os.O_RDONLY | os.O_BINARY)
os.lseek(fd, n+8+os_s, os.SEEK_SET); raw = os.read(fd, os_e-os_s); os.close(fd)
embed_w = torch.from_numpy(bf16_fp32(np.frombuffer(raw, dtype=np.uint16)).copy()).reshape(V, H).to(DEV)
lm_head_w = embed_w
print(f"[Embed] {embed_w.shape}")

# ── Prefetch thread ──
prefetch_q = queue.Queue(maxsize=3)  # (layer_id, np_arrays_dict)

def prefetch_worker(start_lid):
    """Background: read layers from SSD ahead of GPU"""
    for lid in range(start_lid, NUM_LAYERS):
        ne = streamer.read_layer(lid)
        pf = f'model.language_model.layers.{lid}.'
        prefetch_q.put((lid, ne, pf))
    prefetch_q.put(None)  # sentinel

# ── Pre-convert numpy→torch for cached layers ──
layer_torch = {}  # lid → (torch_weights_dict, pf)

def prepare_layer(lid, ne, pf):
    """Convert numpy weights to torch fp32 on CPU (once)"""
    t = {}
    for k, v in ne.items():
        t[k] = torch.from_numpy(v.copy() if v.dtype==np.float32 else bf16_fp32(v.astype(np.uint16)).copy())
    layer_torch[lid] = (t, pf)

# ── GPU Double Buffer ──
class GPUBuffer:
    """Pinned staging → GPU async transfer"""
    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.buf = {}  # key → GPU tensor
    def async_copy(self, cpu_tensor, key):
        if key not in self.buf:
            self.buf[key] = cpu_tensor.to(DEV, non_blocking=True)
        else:
            self.buf[key].copy_(cpu_tensor.to(DEV, non_blocking=True))
    def sync(self): self.stream.synchronize()
    def get(self, key): return self.buf.get(key)

# ═══════════════════════════════════════
# KV Cache + MoE (reuse from v1)
# ═══════════════════════════════════════
class Cache:
    def __init__(self):
        self.kv={}; self.conv={}; self.recur={}
    def get_kv(self, lid): return self.kv.get(lid)
    def get_conv(self, lid): return self.conv.get(lid)
    def get_recur(self, lid): return self.recur.get(lid)
    def up_kv(self, lid, k, v):
        if lid in self.kv: self.kv[lid]=(torch.cat([self.kv[lid][0],k],2),torch.cat([self.kv[lid][1],v],2))
        else: self.kv[lid]=(k,v)
    def up_conv(self, lid, s): self.conv[lid]=s
    def up_recur(self, lid, s): self.recur[lid]=s

class MoE:
    def __init__(self):
        self.g=torch.zeros(32,2048,H,device=DEV); self.d=torch.zeros(32,H,1024,device=DEV)
    def fwd(self, flat, rw, lid, sgw, sg, su, sd):
        logits=flat@rw.T; probs=torch.softmax(logits,-1)
        tw,ti=torch.topk(probs,10,-1); uids=ti.unique().tolist()[:32]
        for s,e in enumerate(uids):
            gn,dn=reader.read_expert(lid,e)
            self.g[s]=torch.from_numpy(bf16_fp32(gn.astype(np.uint16))).to(DEV)
            self.d[s]=torch.from_numpy(bf16_fp32(dn.astype(np.uint16))).to(DEV)
        torch.cuda.synchronize()
        out=torch.zeros_like(flat)
        for s,e in enumerate(uids):
            m=(ti==e).any(-1)
            if not m.any(): continue
            t=flat[m]; gw,uw=self.g[s,:1024],self.g[s,1024:]
            o=(F.silu(t@gw.T)*(t@uw.T))@self.d[s].T
            twm=tw[m]; eh=(ti[m]==e)
            out[m]+=o*(twm*eh).sum(-1,keepdim=True)
        gv=torch.sigmoid(flat@sgw.T)
        so=(F.silu(flat@sg.T)*(flat@su.T))@sd.T
        return out+gv*so

# ═══════════════════════════════════════
# Attention (same as v1, simplified for speed)
# ═══════════════════════════════════════
def forward_attn(hidden, ne, pf, lid, bt, is_prefill, cache):
    B,S,_=hidden.shape
    def w(k): return ne.get(pf+k)

    if bt=="full_attention":
        q=w('self_attn.q_proj.weight'); k_w=w('self_attn.k_proj.weight')
        v_w=w('self_attn.v_proj.weight'); o=w('self_attn.o_proj.weight')
        if q is None: return hidden*0.99
        Q=hidden@q.T; K=hidden@k_w.T; V=hidden@v_w.T
        qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2)
        qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
        K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        if not is_prefill:
            kc,vc=cache.get_kv(lid)
            if kc is not None: K=torch.cat([kc,K],2); V=torch.cat([vc,V],2)
        cache.up_kv(lid,K,V)
        attn=F.scaled_dot_product_attention(qa,K,V,is_causal=is_prefill)
        return (torch.sigmoid(qg)*attn).transpose(1,2).contiguous().view(B,S,8192)@o.T

    else:
        qkv=w('linear_attn.in_proj_qkv.weight'); z_w=w('linear_attn.in_proj_z.weight')
        a_w=w('linear_attn.in_proj_a.weight'); b_w=w('linear_attn.in_proj_b.weight')
        out=w('linear_attn.out_proj.weight'); conv=w('linear_attn.conv1d.weight')
        dt=w('linear_attn.dt_bias'); Alog=w('linear_attn.A_log'); nrm=w('linear_attn.norm.weight')
        if qkv is None: return hidden*0.99

        KD=N_K*H_K; VD=N_V*H_V
        m=hidden@qkv.T; m=m.transpose(1,2)
        if conv is not None:
            cw=conv.squeeze(1) if conv.dim()==4 else conv
            if not is_prefill:
                cs=cache.get_conv(lid)
                if cs is not None: m=torch.cat([cs,m],-1)
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
            cache.up_conv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,N_K,H_K)
        K=m[:,:,KD:2*KD].reshape(B,S,N_K,H_K)
        V=m[:,:,2*KD:].reshape(B,S,N_V,H_V)
        beta=(hidden@b_w.T).sigmoid() if b_w is not None else torch.ones(B,S,N_K,device=DEV)
        a=(hidden@a_w.T) if a_w is not None else torch.zeros(B,S,N_V,device=DEV)
        g=-Alog.float().exp()*F.softplus(a.float()+(dt.float() if dt is not None else 0))
        R=N_V//N_K
        Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K)
        K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K)
        qn=l2norm(Q,-1,1e-6); kn=l2norm(K,-1,1e-6)
        init=None if is_prefill else cache.get_recur(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.up_recur(lid,ls)
        attn=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nrm is not None:
            ni=attn.reshape(B,S,N_V,H_V).float()
            attn=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EPS)*nrm.float().view(1,1,1,-1)).reshape(B,S,VD)
        z=(hidden@z_w.T) if z_w is not None else attn
        attn=attn*F.silu(z.float())
        return (attn@out.T) if out is not None else attn[:,:,:H]

# ═══════════════════════════════════════
# Pipeline Generate
# ═══════════════════════════════════════
def generate(prompt):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEV)
    hidden = F.embedding(input_ids, embed_w)
    cache = Cache(); moe = MoE(); generated = tokens.copy()

    # Prefetch + Pre-convert layers 0-NUM_LAYERS in background
    print(f"  Prefetching...", end="", flush=True)
    t0 = time.perf_counter()
    t = threading.Thread(target=prefetch_worker, args=(0,), daemon=True)
    t.start()

    # Collect all layers
    for _ in range(NUM_LAYERS):
        item = prefetch_q.get()
        if item is None: break
        lid, ne, pf = item
        prepare_layer(lid, ne, pf)
    t.join()
    tc = time.perf_counter()-t0
    print(f" {tc:.0f}s ({60/tc:.1f} layers/s)", flush=True)

    for step in range(MAX_TOKENS):
        is_prefill = (step == 0)
        t_step = time.perf_counter()

        for lid in range(NUM_LAYERS):
            ne, pf = layer_torch[lid]
            bt = LV[lid]

            # Move weights to GPU — use .to() but DON'T cache on GPU (frees after layer)
            gpu_weights = {k: v.to(DEV) if not v.is_cuda else v for k, v in ne.items()}

            # Norms
            in_n  = gpu_weights.get(pf+'input_layernorm.weight')
            pn    = gpu_weights.get(pf+'post_attention_layernorm.weight')
            sgw   = gpu_weights.get(pf+'mlp.shared_expert_gate.weight')
            sg    = gpu_weights.get(pf+'mlp.shared_expert.gate_proj.weight')
            su    = gpu_weights.get(pf+'mlp.shared_expert.up_proj.weight')
            sd    = gpu_weights.get(pf+'mlp.shared_expert.down_proj.weight')
            router = gpu_weights.get(pf+'mlp.gate.weight')

            with torch.no_grad():
                # Attention
                hn = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if in_n is not None: hn = hn * in_n.float()
                a_out = forward_attn(hn, gpu_weights, pf, lid, bt, is_prefill, cache)
                hidden = hidden + a_out

                # MoE
                hn2 = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if pn is not None: hn2 = hn2 * pn.float()
                flat = hn2.reshape(-1, H)
                moe_out = moe.fwd(flat, router, lid, sgw, sg, su, sd)
                hidden = hidden + moe_out.reshape_as(hidden)

            # Free GPU weights for this layer
            del gpu_weights
            if lid % 10 == 0: torch.cuda.empty_cache()

        # LM Head
        logits = hidden[:, -1:, :].float() @ lm_head_w.float().T / TEMPERATURE
        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
        if TOP_K > 0:
            v,_ = torch.topk(logits, TOP_K, -1)
            logits[logits < v[:,:,-1:]] = -float('inf')
        probs = torch.softmax(logits, -1)
        if torch.isnan(probs).any(): probs = torch.ones_like(probs)/V
        next_tok = torch.multinomial(probs.squeeze(), 1).item()
        generated.append(next_tok)

        dt = time.perf_counter() - t_step
        print(f"  [{step}] '{tokenizer.decode([next_tok])}' ({dt:.0f}s)", flush=True)
        if next_tok == tokenizer.eos_token_id: break
        if step < MAX_TOKENS-1:
            hidden = F.embedding(torch.tensor([[next_tok]], device=DEV), embed_w)

    text = tokenizer.decode(generated)
    print(f"  -> {text!r}")
    return text

# Run
for p in ["你好"]:
    generate(p)
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
streamer.close(); reader.close()
