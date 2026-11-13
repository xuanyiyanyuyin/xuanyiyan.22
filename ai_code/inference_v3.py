"""
Qwen3.5-397B 推理 v3 — 多线程I/O + GPU只放计算图
i5-14600KF 20线程 → 16线程拉SSD, GPU 10GB预算, 算完RAM→VRAM→Pagefile
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct
import threading, queue
from concurrent.futures import ThreadPoolExecutor, Future
from collections import OrderedDict

# ═══════════ CONFIG ═══════════
MODEL_DIR = r'D:\ai\models\qwen3.5-397b-bf16'
DEV = "cuda"
IO_THREADS = 16
MAX_TOKENS = 5
TEMPERATURE = 0.8
TOP_K = 40

sys.path.insert(0, r'D:\ai')
from expert_reader import SafeExpertReader
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, l2norm
)

# ═══════════ INIT ═══════════
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
tc = cfg.text_config
reader = SafeExpertReader(MODEL_DIR)
io_pool = ThreadPoolExecutor(max_workers=IO_THREADS)

H = tc.hidden_size; V = tc.vocab_size; LV = tc.layer_types
NV=64; NK=16; HK=128; HV=128; EPS=tc.rms_norm_eps
B2F = lambda a: a.float().bitwise_left_shift(16).view(torch.float32) if a.dtype==torch.uint16 else a

print(f"[v3] {IO_THREADS} IO threads | GPU compute-only | RAM→VRAM→Pagefile eviction")

# ═══════════ EMBED (GPU resident, small) ═══════════
with open(os.path.join(MODEL_DIR,'model.safetensors.index.json')) as f:
    wm = json.load(f)['weight_map']
for k in wm:
    if 'embed_tokens' in k and 'language_model' in k: ek=k; break
with open(os.path.join(MODEL_DIR, wm[ek]), 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
m = hdr[ek]; s,e = m['data_offsets']
fd = os.open(os.path.join(MODEL_DIR, wm[ek]), os.O_RDONLY|os.O_BINARY)
os.lseek(fd, n+8+s, os.SEEK_SET); raw = os.read(fd, e-s); os.close(fd)
emb = torch.from_numpy(np.frombuffer(raw,dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV)

# ═══════════ RAM CACHE + PREFETCH ═══════════
from non_expert_streamer import NonExpertStreamer
streamer = NonExpertStreamer(MODEL_DIR)
layer_cache = {}  # lid → {key: cpu_tensor fp32}
print(f"Prefetching 60 layers to RAM...", end="", flush=True)
t0 = time.perf_counter()
for lid in range(60):
    ne = streamer.read_layer(lid)
    t = {}
    for k,v in ne.items():
        if v.dtype == np.float32:
            t[k] = torch.from_numpy(v.copy())
        else:
            t[k] = torch.from_numpy(v.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16).float()
    layer_cache[lid] = t
print(f" {time.perf_counter()-t0:.0f}s")

# ═══════════ EXPERT PREFETCH SYSTEM ═══════════
expert_ram = {}  # (lid, eid) → (gate_up_cpu, down_cpu)
prefetch_q = queue.Queue(maxsize=200)  # (lid, eid_list) to prefetch

def prefetch_worker():
    while True:
        item = prefetch_q.get()
        if item is None: break
        lid, eids = item
        for eid in eids:
            if (lid, eid) not in expert_ram:
                gn, dn = reader.read_expert(lid, eid)
                expert_ram[(lid,eid)] = (
                    torch.from_numpy(gn.astype(np.int16).copy()).view(torch.bfloat16),
                    torch.from_numpy(dn.astype(np.int16).copy()).view(torch.bfloat16)
                )

# Start IO threads
io_threads = [threading.Thread(target=prefetch_worker, daemon=True) for _ in range(IO_THREADS)]
for t in io_threads: t.start()

# ═══════════ KV CACHE ═══════════
class Cache:
    def __init__(self):
        self.kv={}; self.conv={}; self.recur={}
    def get_kv(self,lid): return self.kv.get(lid)
    def get_conv(self,lid): return self.conv.get(lid)
    def get_recur(self,lid): return self.recur.get(lid)
    def up_kv(self,lid,k,v):
        if lid in self.kv: self.kv[lid]=(torch.cat([self.kv[lid][0],k],2),torch.cat([self.kv[lid][1],v],2))
        else: self.kv[lid]=(k,v)
    def up_conv(self,lid,s): self.conv[lid]=s
    def up_recur(self,lid,s): self.recur[lid]=s

# ═══════════ ATTENTION (same as before, compact) ═══════════
def forward_attn(hidden, ne_gpu, lid, bt, is_prefill, cache):
    B,S,_=hidden.shape; pf=f'model.language_model.layers.{lid}.'
    def w(k): v=ne_gpu.get(pf+k); return v.float() if v is not None else None
    if bt=="full_attention":
        q=w('self_attn.q_proj.weight'); k_w=w('self_attn.k_proj.weight')
        v_w=w('self_attn.v_proj.weight'); o=w('self_attn.o_proj.weight')
        if q is None: return hidden*0.99
        Q=hidden.float()@q.T; K=hidden.float()@k_w.T; V=hidden.float()@v_w.T
        qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2); qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2)
        K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1); V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
        if not is_prefill: kc,vc=cache.get_kv(lid); K=torch.cat([kc,K],2) if kc is not None else K; V=torch.cat([vc,V],2) if vc is not None else V
        cache.up_kv(lid,K,V)
        attn=F.scaled_dot_product_attention(qa,K,V,is_causal=is_prefill)
        return ((torch.sigmoid(qg)*attn).transpose(1,2).contiguous().view(B,S,8192)@o.T).to(torch.bfloat16)
    else:
        qkv=w('linear_attn.in_proj_qkv.weight'); z_w=w('linear_attn.in_proj_z.weight')
        a_w=w('linear_attn.in_proj_a.weight'); b_w=w('linear_attn.in_proj_b.weight')
        out=w('linear_attn.out_proj.weight'); conv=w('linear_attn.conv1d.weight')
        dt=w('linear_attn.dt_bias'); Alog=w('linear_attn.A_log'); nrm=w('linear_attn.norm.weight')
        if qkv is None: return hidden*0.99
        K_S=NK*HK; V_S=NV*HV
        m=hidden.float()@qkv.T; m=m.transpose(1,2)
        if conv is not None:
            cw=conv.squeeze(1) if conv.dim()==4 else conv
            if not is_prefill: cs=cache.get_conv(lid); m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]; cache.up_conv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:K_S].reshape(B,S,NK,HK); K=m[:,:,K_S:2*K_S].reshape(B,S,NK,HK); V=m[:,:,2*K_S:].reshape(B,S,NV,HV)
        beta=(hidden.float()@b_w.T).sigmoid() if b_w is not None else torch.ones(B,S,NK,device=DEV)
        a=(hidden.float()@a_w.T) if a_w is not None else torch.zeros(B,S,NV,device=DEV)
        g=-Alog.float().exp()*F.softplus(a+(dt.float() if dt is not None else 0))
        R=NV//NK; Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK); K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
        qn=l2norm(Q,-1,1e-6); kn=l2norm(K,-1,1e-6)
        init=None if is_prefill else cache.get_recur(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.up_recur(lid,ls)
        attn=ca.float().transpose(1,2).contiguous().view(B,S,V_S)
        if nrm is not None: ni=attn.reshape(B,S,NV,HV); attn=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EPS)*nrm.view(1,1,1,-1)).reshape(B,S,V_S)
        z=(hidden.float()@z_w.T) if z_w is not None else attn; attn=attn*F.silu(z)
        return ((attn@out.T) if out is not None else attn[:,:,:H]).to(torch.bfloat16)

# ═══════════ MOE (per-expert serial, uses prefetched RAM) ═══════════
def moe_forward(flat, router_w, lid, sgw, sg, su, sd):
    logits = flat.float() @ router_w.float().T
    probs = torch.softmax(logits, -1)
    _, ti = torch.topk(probs, 10, -1)
    uids = ti.unique().tolist()[:10]

    out = torch.zeros_like(flat, dtype=torch.float32)
    for eid in uids:
        key = (lid, eid)
        if key not in expert_ram:
            gn, dn = reader.read_expert(lid, eid)
            expert_ram[key] = (
                torch.from_numpy(gn.astype(np.int16).copy()).view(torch.bfloat16),
                torch.from_numpy(dn.astype(np.int16).copy()).view(torch.bfloat16)
            )
        gu_cpu, dn_cpu = expert_ram[key]
        gu = gu_cpu.float().to(DEV); dn = dn_cpu.float().to(DEV)
        mask = (ti == eid).any(-1)
        if not mask.any(): del gu, dn; continue
        t = flat[mask].float()
        o = (F.silu(t @ gu[:1024].T) * (t @ gu[1024:].T)) @ dn.T
        tw = probs[mask]; eh = (ti[mask] == eid)
        out[mask] += o * (tw * eh).sum(-1, keepdim=True)
        del gu, dn

    gv = torch.sigmoid(flat.float() @ sgw.float().T)
    so = (F.silu(flat.float() @ sg.float().T) * (flat.float() @ su.float().T)) @ sd.float().T
    return (out + gv * so).to(torch.bfloat16)

# ═══════════ GENERATE ═══════════
def generate(prompt):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEV)
    hidden = F.embedding(input_ids, emb)
    cache = Cache()
    gen = tokens.copy()

    for step in range(MAX_TOKENS):
        is_prefill = (step == 0)
        t0 = time.perf_counter()

        # Prefetch experts for ALL layers (fire-and-forget in background)
        for lid in range(60):
            prefetch_q.put((lid, list(range(20))))  # top-20 most likely

        for lid in range(60):
            ne_cpu = layer_cache[lid]; bt = LV[lid]; pf = f'model.language_model.layers.{lid}.'

            # Move only needed weights to GPU
            ne = {}
            for k_suffix in ['input_layernorm.weight','post_attention_layernorm.weight',
                'mlp.shared_expert_gate.weight','mlp.shared_expert.gate_proj.weight',
                'mlp.shared_expert.up_proj.weight','mlp.shared_expert.down_proj.weight',
                'mlp.gate.weight',
                'self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight',
                'linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight',
                'linear_attn.in_proj_a.weight','linear_attn.in_proj_b.weight',
                'linear_attn.out_proj.weight','linear_attn.conv1d.weight',
                'linear_attn.dt_bias','linear_attn.A_log','linear_attn.norm.weight']:
                v = ne_cpu.get(pf+k_suffix)
                if v is not None: ne[pf+k_suffix] = v.to(DEV, non_blocking=True)

            in_n = ne.get(pf+'input_layernorm.weight')
            pn   = ne.get(pf+'post_attention_layernorm.weight')
            router = ne.get(pf+'mlp.gate.weight')
            sgw  = ne.get(pf+'mlp.shared_expert_gate.weight')
            sg   = ne.get(pf+'mlp.shared_expert.gate_proj.weight')
            su   = ne.get(pf+'mlp.shared_expert.up_proj.weight')
            sd   = ne.get(pf+'mlp.shared_expert.down_proj.weight')

            with torch.no_grad():
                hn = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if in_n is not None: hn = hn * in_n.float()
                hn = hn.to(torch.bfloat16)
                a_out = forward_attn(hn, ne, lid, bt, is_prefill, cache)
                hidden = (hidden + a_out).to(torch.bfloat16)

                hn2 = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if pn is not None: hn2 = hn2 * pn.float()
                flat = hn2.to(torch.bfloat16).reshape(-1, H)
                moe_out = moe_forward(flat, router, lid, sgw, sg, su, sd)
                hidden = (hidden + moe_out.reshape_as(hidden)).to(torch.bfloat16)

            # Evict: GPU → CPU → pagefile (just del, OS handles rest)
            for k in list(ne.keys()): del ne[k]
            del ne
            if lid % 15 == 0: torch.cuda.empty_cache()

        # ── LM Head ──
        logits = hidden[:, -1:, :].float() @ emb.float().T / TEMPERATURE
        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
        if TOP_K > 0: v,_ = torch.topk(logits, TOP_K, -1); logits[logits < v[:,:,-1:]] = -float('inf')
        probs = torch.softmax(logits, -1)
        if torch.isnan(probs).any(): probs = torch.ones_like(probs)/V
        next_tok = torch.multinomial(probs.squeeze(), 1).item()
        gen.append(next_tok)
        dt = time.perf_counter() - t0
        print(f"  [{step}] '{tokenizer.decode([next_tok])}' ({dt:.0f}s)", flush=True)
        if next_tok == tokenizer.eos_token_id: break
        if step < MAX_TOKENS-1:
            hidden = F.embedding(torch.tensor([[next_tok]], device=DEV), emb).to(torch.bfloat16)

    return tokenizer.decode(gen)

# ═══════════ RUN ═══════════
for p in ["你好"]:
    print(f"\n{'─'*40}\n{p!r}\n{'─'*40}")
    text = generate(p)
    print(f"  → {text!r}")

for _ in range(IO_THREADS): prefetch_q.put(None)
for t in io_threads: t.join(timeout=1)
io_pool.shutdown(wait=False)
streamer.close(); reader.close()
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
