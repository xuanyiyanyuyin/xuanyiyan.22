"""
397B推理 × V22 Stage E 读写头
FullAttention: KV Cache → 文件记忆检索
DeltaNet:      Recurrent State → 文件记忆读写
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, os, shutil
sys.path.insert(0, r'D:\ai')
from memory_head import MemoryHead

DEV="cuda"; H=4096; B=1
STORE = r"D:\ai\memory_store"
shutil.rmtree(STORE, ignore_errors=True)

# ═══════════ Memory Heads (one per layer) ═══════════
# FullAttention: 15 layers, each has own memory store
kv_memory = {lid: MemoryHead(f"{STORE}/kv_L{lid:02d}", max_gpu=200, max_ram=2000)
             for lid in range(60) if lid % 4 == 3}
# DeltaNet: 45 layers, each has own recurrent memory
rc_memory = {lid: MemoryHead(f"{STORE}/rc_L{lid:02d}", max_gpu=300, max_ram=3000)
             for lid in range(60) if lid % 4 != 3}

print(f"[Memory] KV:{len(kv_memory)} layers, RC:{len(rc_memory)} layers")

# ═══════════ Virtual Weights (shared templates) ═══════════
class FullW:
    def __init__(self):
        self.n=torch.randn(H,device=DEV);self.q=torch.randn(16384,H,device=DEV)
        self.k=torch.randn(512,H,device=DEV);self.v=torch.randn(512,H,device=DEV)
        self.o=torch.randn(4096,8192,device=DEV)
        self.gu=torch.randn(10,2048,H,device=DEV);self.dn=torch.randn(10,H,1024,device=DEV)
        self.rw=torch.randn(512,H,device=DEV);self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV);self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)
class DeltaW:
    def __init__(self):
        self.n=torch.randn(H,device=DEV);self.w1=torch.randn(H,H,device=DEV)
        self.w2=torch.randn(H,H,device=DEV)
        self.gu=torch.randn(10,2048,H,device=DEV);self.dn=torch.randn(10,H,1024,device=DEV)
        self.rw=torch.randn(512,H,device=DEV);self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV);self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)

FULL = FullW(); DELTA = DeltaW()
print(f"[Weights] shared templates, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

# ═══════════ Forward with Memory Head ═══════════
def forward_full_attn(h, l, lid, is_prefill):
    """FullAttention + V22 Memory retrieval"""
    # Memory READ: query file-based memory instead of GPU KV cache
    recall = kv_memory[lid].read(h, layer_id=lid, top_k=2, mem_type='full_attn')

    # Standard attention
    hn = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) * l.n
    Q = h.float() @ l.q.T; K = h.float() @ l.k.T; V = h.float() @ l.v.T
    B, S, _ = h.shape
    qa = Q[:, :, :8192].view(B, S, 32, 256).transpose(1, 2)
    qg = Q[:, :, 8192:].view(B, S, 32, 256).transpose(1, 2)
    K = K.view(B, S, 2, 256).transpose(1, 2).repeat_interleave(16, 1)
    V = V.view(B, S, 2, 256).transpose(1, 2).repeat_interleave(16, 1)

    # Inject recalled memory into KV (spherical fusion)
    if recall is not None and recall.shape[-1] >= 1024:
        rk = recall[:, :512].view(1, 1, 2, 256).repeat_interleave(16, 1).to(DEV)
        rv = recall[:, 512:1024].view(1, 1, 2, 256).repeat_interleave(16, 1).to(DEV)
        K = torch.cat([K, rk], dim=2)
        V = torch.cat([V, rv], dim=2)

    att = F.scaled_dot_product_attention(qa, K, V, is_causal=is_prefill)
    a_out = ((torch.sigmoid(qg) * att).transpose(1, 2)
             .contiguous().view(B, S, 8192) @ l.o.T).to(torch.bfloat16)
    h = (h + a_out).to(torch.bfloat16)

    # Memory WRITE: save current K/V projection to file
    kv_memory[lid].write(h, layer_id=lid, mem_type='full_attn')
    return h


def forward_delta_net(h, l, lid, is_prefill):
    """DeltaNet simplified + V22 Memory retrieval"""
    recall = rc_memory[lid].read(h, layer_id=lid, top_k=3, mem_type='delta')

    hn = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) * l.n
    a_out = (F.silu(h.float() @ l.w1.T) @ l.w2.T).to(torch.bfloat16)
    h = (h + a_out).to(torch.bfloat16)

    rc_memory[lid].write(h, layer_id=lid, mem_type='delta')
    return h


def forward_moe(h, l):
    """MoE (unchanged)"""
    flat = h.reshape(-1, H).float()
    _logits = flat @ l.rw.T
    _ = torch.topk(torch.softmax(_logits, -1), 10, -1)
    for e in range(10):
        gw, uw = l.gu[e, :1024], l.gu[e, 1024:]
        gate = F.silu(flat @ gw.T); up = flat @ uw.T
        _moe = (gate * up) @ l.dn[e].T
    gv = torch.sigmoid(flat @ l.sgw.T)
    so = (F.silu(flat @ l.sg.T) * (flat @ l.su.T)) @ l.sd.T
    return h + (gv * so).to(torch.bfloat16).reshape_as(h)

# ═══════════ Run ═══════════
print("\n=== V22 Memory Head × 397B ===")
h = torch.randn(B, 8, H, dtype=torch.bfloat16, device=DEV)

for step in range(5):
    is_prefill = (step == 0)
    t0 = time.perf_counter()

    for lid in range(60):
        if lid % 4 == 3:
            h = forward_full_attn(h, FULL, lid, is_prefill)
        else:
            h = forward_delta_net(h, DELTA, lid, is_prefill)
        h = forward_moe(h, FULL if lid%4==3 else DELTA)

    elapsed = time.perf_counter() - t0
    kv_stats = next(iter(kv_memory.values())).stats()
    rc_stats = next(iter(rc_memory.values())).stats()
    print(f"  [{step}] {elapsed:.1f}s | KV:{kv_stats} | RC:{rc_stats}")

# Show storage
for name, mem_dict in [("KV", kv_memory), ("RC", rc_memory)]:
    total_files = sum(len(os.listdir(f"{m.store_dir}/{m.store_dir.split('/')[-1]}"))
                      for m in mem_dict.values()
                      if os.path.exists(f"{m.store_dir}/{m.store_dir.split('/')[-1]}"))
    print(f"{name}: {total_files} files across {len(mem_dict)} layers")

shutil.rmtree(STORE, ignore_errors=True)
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
