"""
Qwen3.5-397B 流式推理 + 多线程并行SSD读取
策略: GPU算Attention时, 后台线程并发读Expert
"""
import torch, torch.nn.functional as F, time, numpy as np
import sys, json, os, struct, threading, queue
from concurrent.futures import ThreadPoolExecutor

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
MAX_TOKENS=5; TEMPERATURE=0.8; TOP_K=40
NUM_THREADS=8

print(f"[Threaded] {NUM_THREADS} threads for SSD reads")

# ── Embed ──
def bf16_fp32(a): return (a.astype(np.uint32) << 16).view(np.float32)
with open(os.path.join(MODEL_DIR,'model.safetensors.index.json')) as f:
    wm = json.load(f)['weight_map']
for k in wm:
    if 'embed_tokens' in k and 'language_model' in k: ek=k; break
with open(os.path.join(MODEL_DIR, wm[ek]), 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
m = hdr[ek]; os_s, os_e = m['data_offsets']
fd = os.open(os.path.join(MODEL_DIR, wm[ek]), os.O_RDONLY|os.O_BINARY)
os.lseek(fd, n+8+os_s, os.SEEK_SET); raw = os.read(fd, os_e-os_s); os.close(fd)
embed_w = torch.from_numpy(bf16_fp32(np.frombuffer(raw, dtype=np.uint16)).copy()).reshape(V,H).to(DEV)
print(f"[Embed] {V}x{H} on GPU")

# ── RAM Cache + Prefetch ──
layer_cpu = []  # (lid, {key: cpu_tensor})
print(f"Prefetching 60 layers...", end="", flush=True)
t0 = time.perf_counter()
for lid in range(60):
    ne = streamer.read_layer(lid)
    t = {}
    for k,v in ne.items():
        t[k] = torch.from_numpy(v.copy() if v.dtype==np.float32 else bf16_fp32(v.astype(np.uint16)).copy())
    layer_cpu.append((lid, t))
print(f" {time.perf_counter()-t0:.0f}s")

# ── Threaded Expert Reader ──
expert_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)
expert_futures = {}  # (lid, eid) → Future

def read_expert_async(lid, eid):
    """Read expert in background thread, return (gpu_tensor_gateup, gpu_tensor_down)"""
    gn, dn = reader.read_expert(lid, eid)
    gu = torch.from_numpy(bf16_fp32(gn.astype(np.uint16)).copy())
    dw = torch.from_numpy(bf16_fp32(dn.astype(np.uint16)).copy())
    return gu, dw

# ── KV Cache ──
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

# ── Attention ──
def forward_attn(hidden, ne, lid, bt, is_prefill, cache):
    B,S,_=hidden.shape; pf=f'model.language_model.layers.{lid}.'
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
        if not is_prefill: kc,vc=cache.get_kv(lid); K=torch.cat([kc,K],2) if kc is not None else K; V=torch.cat([vc,V],2) if vc is not None else V
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
            if not is_prefill: cs=cache.get_conv(lid); m=torch.cat([cs,m],-1) if cs is not None else m
            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
            cache.up_conv(lid,m[:,:,-3:].clone())
        m=m.transpose(1,2)
        Q=m[:,:,:KD].reshape(B,S,N_K,H_K); K=m[:,:,KD:2*KD].reshape(B,S,N_K,H_K); V=m[:,:,2*KD:].reshape(B,S,N_V,H_V)
        beta=(hidden@b_w.T).sigmoid() if b_w is not None else torch.ones(B,S,N_K,device=DEV)
        a=(hidden@a_w.T) if a_w is not None else torch.zeros(B,S,N_V,device=DEV)
        g=-Alog.float().exp()*F.softplus(a.float()+(dt.float() if dt is not None else 0))
        R=N_V//N_K; Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K); K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,N_V,H_K)
        qn=l2norm(Q,-1,1e-6); kn=l2norm(K,-1,1e-6)
        init=None if is_prefill else cache.get_recur(lid)
        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
        cache.up_recur(lid,ls)
        attn=ca.float().transpose(1,2).contiguous().view(B,S,VD)
        if nrm is not None: ni=attn.reshape(B,S,N_V,H_V).float(); attn=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EPS)*nrm.float().view(1,1,1,-1)).reshape(B,S,VD)
        z=(hidden@z_w.T) if z_w is not None else attn; attn=attn*F.silu(z.float())
        return (attn@out.T) if out is not None else attn[:,:,:H]

# ── Generate ──
def generate(prompt):
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEV)
    hidden = F.embedding(input_ids, embed_w)
    cache = Cache(); generated = tokens.copy()

    for step in range(MAX_TOKENS):
        is_prefill = (step == 0)
        t_step = time.perf_counter()

        for lid in range(60):
            _, ne_cpu = layer_cpu[lid]
            bt = LV[lid]; pf = f'model.language_model.layers.{lid}.'

            # Move non-expert weights to GPU (only once per step, then free)
            ne = {k: v.to(DEV) for k, v in ne_cpu.items()}
            in_n = ne.get(pf+'input_layernorm.weight')
            pn   = ne.get(pf+'post_attention_layernorm.weight')
            sgw  = ne.get(pf+'mlp.shared_expert_gate.weight')
            sg   = ne.get(pf+'mlp.shared_expert.gate_proj.weight')
            su   = ne.get(pf+'mlp.shared_expert.up_proj.weight')
            sd_  = ne.get(pf+'mlp.shared_expert.down_proj.weight')
            router = ne.get(pf+'mlp.gate.weight')

            # ── Fire expert reads in parallel (background) ──
            # Read 20 most-likely experts for this layer
            for eid in range(min(20, 512)):
                expert_futures[(lid, eid)] = expert_pool.submit(read_expert_async, lid, eid)

            with torch.no_grad():
                hn = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if in_n is not None: hn = hn * in_n.float()
                a_out = forward_attn(hn, ne, lid, bt, is_prefill, cache)
                hidden = hidden + a_out

                # Router: determine which experts
                hn2 = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1,keepdim=True)+EPS)
                if pn is not None: hn2 = hn2 * pn.float()
                flat = hn2.reshape(-1, H)
                logits = flat @ router.T
                probs = torch.softmax(logits, -1)
                _, ti = torch.topk(probs, 10, -1)
                uids = ti.unique().tolist()[:10]

                # ── Collect experts from background threads ──
                gu_list, dn_list = [], []
                for eid in uids:
                    gu, dn = expert_futures.pop((lid, eid), None) or read_expert_async(lid, eid)
                    gu_list.append(gu.to(DEV)); dn_list.append(dn.to(DEV))

                # MoE compute
                out_moe = torch.zeros_like(flat)
                for s, eid in enumerate(uids):
                    gu_w = gu_list[s]; dn_w = dn_list[s]
                    mask = (ti == eid).any(-1)
                    if not mask.any(): continue
                    t = flat[mask]
                    gate_w, up_w = gu_w[:1024], gu_w[1024:]
                    o = (F.silu(t @ gate_w.T) * (t @ up_w.T)) @ dn_w.T
                    tw = probs[mask]; eh = (ti[mask] == eid)
                    out_moe[mask] += o * (tw * eh).sum(-1, keepdim=True)

                gv = torch.sigmoid(flat @ sgw.T)
                so = (F.silu(flat @ sg.T) * (flat @ su.T)) @ sd_.T
                hidden = hidden + (out_moe + gv * so).reshape_as(hidden)

            # Free GPU
            for k in list(ne.keys()): del ne[k]
            del ne, gu_list, dn_list, out_moe
            if lid % 20 == 0: torch.cuda.empty_cache()

        logits = hidden[:, -1:, :].float() @ embed_w.float().T / TEMPERATURE
        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=-1e9)
        if TOP_K > 0: v,_ = torch.topk(logits, TOP_K, -1); logits[logits < v[:,:,-1:]] = -float('inf')
        probs = torch.softmax(logits, -1)
        if torch.isnan(probs).any(): probs = torch.ones_like(probs)/V
        next_tok = torch.multinomial(probs.squeeze(), 1).item()
        generated.append(next_tok)
        dt = time.perf_counter() - t_step
        print(f"  [{step}] '{tokenizer.decode([next_tok])}' {dt:.0f}s", flush=True)
        if next_tok == tokenizer.eos_token_id: break
        if step < MAX_TOKENS-1:
            hidden = F.embedding(torch.tensor([[next_tok]], device=DEV), embed_w)

    print(f"  -> {tokenizer.decode(generated)!r}")
    return tokenizer.decode(generated)

# Run
for p in ["你好"]:
    generate(p)
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
streamer.close(); reader.close(); expert_pool.shutdown()
