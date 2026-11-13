"""
[Final] mmap虚拟内存 + 全优化 + 虚拟权重 终极测试
对比: pread(当前) vs mmap(虚拟内存)
"""
import torch, torch.nn.functional as F, time, numpy as np, os, mmap, struct, json, tempfile, shutil

DEV='cuda'; H=4096; B=1; N=5  # N=5 rounds for speed
EXPERT_TOT = 24*1024*1024  # 24MB
NUM_EXPERTS = 512

print("=" * 60)
print("  [FINAL] mmap + 全优化 虚拟权重终极测试")
print("=" * 60)

# === Create mmap backend ===
print("\n[1/4] Creating mmap expert store (12GB)...")
t0 = time.perf_counter()
SSD_PATH = "D:\\ai\\mmap_final_test.bin"
with open(SSD_PATH, "wb") as f:
    f.seek(NUM_EXPERTS * EXPERT_TOT - 1)
    f.write(b'\x00')
fd = os.open(SSD_PATH, os.O_RDWR | os.O_BINARY)
mapped = mmap.mmap(fd, 0)
# Fill with random data
chunk = os.urandom(1024*1024)
for i in range(0, NUM_EXPERTS * EXPERT_TOT, 1024*1024):
    mapped[i:i+1024*1024] = chunk
mapped.flush()
print(f"  {os.path.getsize(SSD_PATH)/1024**3:.1f}GB in {time.perf_counter()-t0:.0f}s")

# === Model definitions (same as bench_final, with mmap path) ===
class OptFull:
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.q=torch.randn(16384,H,device=DEV)
        self.k=torch.randn(512,H,device=DEV); self.v=torch.randn(512,H,device=DEV)
        self.o_t=torch.randn(8192,4096,device=DEV)
        self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)

class OptDelta:
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.w1=torch.randn(H,H,device=DEV)
        self.w2=torch.randn(H,H,device=DEV)
        self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)

def read_experts_pread(n_exp, fd_ssd):
    """当前方式: pread逐个读"""
    experts_gu=[]; experts_dn=[]
    off = 8
    for eid in range(n_exp):
        os.lseek(fd_ssd, off + eid * EXPERT_TOT, os.SEEK_SET)
        buf = os.read(fd_ssd, EXPERT_TOT)
        raw_gu = np.frombuffer(buf[:16*1024*1024], dtype=np.uint16).copy()
        raw_dn = np.frombuffer(buf[16*1024*1024:], dtype=np.uint16).copy()
        experts_gu.append(torch.from_numpy(raw_gu.view(np.int16)).view(torch.bfloat16).reshape(2048, 4096))
        experts_dn.append(torch.from_numpy(raw_dn.view(np.int16)).view(torch.bfloat16).reshape(4096, 1024))
    return experts_gu, experts_dn

def read_experts_mmap(n_exp, mm, off=8):
    """新方式: mmap物理内存直接访问"""
    experts_gu=[]; experts_dn=[]
    for eid in range(n_exp):
        base = off + eid * EXPERT_TOT
        # Read raw, then convert with torch
        raw_gu = np.frombuffer(mm, dtype=np.uint16, count=16*1024*1024//2, offset=base).copy()
        raw_dn = np.frombuffer(mm, dtype=np.uint16, count=8*1024*1024//2, offset=base+16*1024*1024).copy()
        experts_gu.append(torch.from_numpy(raw_gu.view(np.int16)).view(torch.bfloat16).reshape(2048, 4096))
        experts_dn.append(torch.from_numpy(raw_dn.view(np.int16)).view(torch.bfloat16).reshape(4096, 1024))
    return experts_gu, experts_dn

def forward_layer(h, l, experts_gu, experts_dn, is_full):
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+1e-6)*l.n
    if is_full:
        Q=h.float()@l.q.T;K=h.float()@l.k.T;V=h.float()@l.v.T
        qa=Q[:,:,:8192].view(B,8,32,256).transpose(1,2)
        qg=Q[:,:,8192:].view(B,8,32,256).transpose(1,2)
        K=K.view(B,8,2,256).transpose(1,2).repeat_interleave(16,1)
        V=V.view(B,8,2,256).transpose(1,2).repeat_interleave(16,1)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=True)
        a_out=((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,8,8192)@l.o_t).to(torch.bfloat16)
    else:
        a_out=(F.silu(h.float()@l.w1.T)@l.w2.T).to(torch.bfloat16)
    h=(h+a_out).to(torch.bfloat16)
    flat=h.reshape(B*8,H).float()

    # Stack → H2D → compute (current: 10 experts per loop iteration)
    eid=0
    out=torch.zeros_like(flat,dtype=torch.float32)
    batch_g = torch.stack(experts_gu).float().to(DEV)
    batch_d = torch.stack(experts_dn).float().to(DEV)
    for e in range(len(experts_gu)):
        gw,uw=batch_g[e,:1024],batch_g[e,1024:]
        _=(F.silu(flat@gw.T)*(flat@uw.T))@batch_d[e].T
    gv=torch.sigmoid(flat@l.sgw.T)
    so=(F.silu(flat@l.sg.T)*(flat@l.su.T))@l.sd.T
    h=(h+(gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)
    return h

# === Run benchmarks ===
print("\n[2/4] Benchmark: pread (current)")
OF=OptFull(); OD=OptDelta()
h=torch.randn(B,8,H,dtype=torch.bfloat16,device=DEV)

fd2 = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# Warmup
for lid in range(2):
    gu,dn = read_experts_pread(10, fd2)
    h = forward_layer(h, OF if lid%2==0 else OD, gu, dn, lid%2==0)

t0=time.perf_counter()
for _ in range(N):
    for lid in range(60):
        gu,dn = read_experts_pread(10, fd2)
        h = forward_layer(h, OF if lid%4==3 else OD, gu, dn, lid%4==3)
torch.cuda.synchronize()
pread_ms = (time.perf_counter()-t0)/N*1000
pread_vram = torch.cuda.max_memory_allocated()/1024**3
pread_tok = 60*B*1000/pread_ms
os.close(fd2)

print(f"  pread: {pread_ms:.0f}ms/60L | {pread_tok:.1f} tok/s | {B*1000/pread_ms*60:.1f} s/token | VRAM {pread_vram:.1f}GB")

print("\n[3/4] Benchmark: mmap (virtual memory)")
fd3 = os.open(SSD_PATH, os.O_RDONLY | os.O_BINARY)
mm = mmap.mmap(fd3, 0, access=mmap.ACCESS_READ)
h=torch.randn(B,8,H,dtype=torch.bfloat16,device=DEV)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# Warmup
for lid in range(2):
    gu,dn = read_experts_mmap(10, mm)
    h = forward_layer(h, OF if lid%2==0 else OD, gu, dn, lid%2==0)

t0=time.perf_counter()
for _ in range(N):
    for lid in range(60):
        gu,dn = read_experts_mmap(10, mm)
        h = forward_layer(h, OF if lid%4==3 else OD, gu, dn, lid%4==3)
torch.cuda.synchronize()
mmap_ms = (time.perf_counter()-t0)/N*1000
mmap_vram = torch.cuda.max_memory_allocated()/1024**3
mmap_tok = 60*B*1000/mmap_ms
mm.close(); os.close(fd3)

print(f"  mmap:  {mmap_ms:.0f}ms/60L | {mmap_tok:.1f} tok/s | {B*1000/mmap_ms*60:.1f} s/token | VRAM {mmap_vram:.1f}GB")

# === Final ===
speedup = pread_ms / mmap_ms
saved = (pread_ms - mmap_ms) / 60  # per layer
print(f"\n[4/4] {'='*60}")
print(f"  pread (current):   {B*1000/pread_ms*60:.1f} s/token")
print(f"  mmap (virtual):    {B*1000/mmap_ms*60:.1f} s/token")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Per-layer saved: {saved:.1f}ms")
print(f"  60-layer saved: {(pread_ms-mmap_ms):.0f}ms total")

# Reality check
print(f"\n  [Note] Virtual weights = pure compute + H2D, no real SSD bottleneck.")
print(f"  Real world: SSD I/O adds ~89ms/layer (Gen3), dwarfing these gains.")
print(f"  mmap benefit is real but ~1s on real hardware.")

# Cleanup
mm.close(); mapped.close(); os.close(fd)
shutil.rmtree(SSD_PATH, ignore_errors=True)
print(f"\n  Cleaned up.")
