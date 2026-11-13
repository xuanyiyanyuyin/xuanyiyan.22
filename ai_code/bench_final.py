"""
397B 最终优化版 · 虚拟权重完整测试
Gate融合 + O转置 + 预转torch + 专家40堆叠 + V22读写头
"""
import torch, torch.nn.functional as F, time, numpy as np
DEV='cuda'; H=4096; B=1; N=20

# ═══════════ Virtual Weights (optimization versions) ═══════════
class OrigFull:  # original
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.q=torch.randn(16384,H,device=DEV)
        self.k=torch.randn(512,H,device=DEV); self.v=torch.randn(512,H,device=DEV)
        self.o=torch.randn(4096,8192,device=DEV)
        self.gate=torch.randn(10,1024,H,device=DEV); self.up=torch.randn(10,1024,H,device=DEV)
        self.dn=torch.randn(10,H,1024,device=DEV); self.rw=torch.randn(512,H,device=DEV)
        self.sgw=torch.randn(1,H,device=DEV); self.sg=torch.randn(1024,H,device=DEV)
        self.su=torch.randn(1024,H,device=DEV); self.sd=torch.randn(H,1024,device=DEV)

class OptFull:  # optimized: Gate+Up fused, O transposed, experts pre-converted CPU tensor
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.q=torch.randn(16384,H,device=DEV)
        self.k=torch.randn(512,H,device=DEV); self.v=torch.randn(512,H,device=DEV)
        self.o_t=torch.randn(8192,4096,device=DEV)  # pre-transposed!
        # Experts: pre-converted torch bf16 CPU (not numpy!)
        self.gu_cpu=[torch.randn(2048,H,dtype=torch.bfloat16) for _ in range(10)]
        self.dn_cpu=[torch.randn(H,1024,dtype=torch.bfloat16) for _ in range(10)]
        self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)

class OrigDelta:
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.w1=torch.randn(H,H,device=DEV)
        self.w2=torch.randn(H,H,device=DEV)
        self.gate=torch.randn(10,1024,H,device=DEV); self.up=torch.randn(10,1024,H,device=DEV)
        self.dn=torch.randn(10,H,1024,device=DEV); self.rw=torch.randn(512,H,device=DEV)
        self.sgw=torch.randn(1,H,device=DEV); self.sg=torch.randn(1024,H,device=DEV)
        self.su=torch.randn(1024,H,device=DEV); self.sd=torch.randn(H,1024,device=DEV)

class OptDelta:
    def __init__(self):
        self.n=torch.randn(H,device=DEV); self.w1=torch.randn(H,H,device=DEV)
        self.w2=torch.randn(H,H,device=DEV)
        self.gu_cpu=[torch.randn(2048,H,dtype=torch.bfloat16) for _ in range(10)]
        self.dn_cpu=[torch.randn(H,1024,dtype=torch.bfloat16) for _ in range(10)]
        self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
        self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
        self.sd=torch.randn(H,1024,device=DEV)

# ═══════════ Forward functions ═══════════

def forward_orig(h, l, is_full):
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+1e-6)*l.n
    if is_full:
        Q=h.float()@l.q.T;K=h.float()@l.k.T;V=h.float()@l.v.T
        qa=Q[:,:,:8192].view(B,8,32,256).transpose(1,2)
        qg=Q[:,:,8192:].view(B,8,32,256).transpose(1,2)
        K=K.view(B,8,2,256).transpose(1,2).repeat_interleave(16,1)
        V=V.view(B,8,2,256).transpose(1,2).repeat_interleave(16,1)
        att=F.scaled_dot_product_attention(qa,K,V,is_causal=True)
        a_out=((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,8,8192)@l.o.T).to(torch.bfloat16)
    else:
        a_out=(F.silu(h.float()@l.w1.T)@l.w2.T).to(torch.bfloat16)
    h=(h+a_out).to(torch.bfloat16)
    flat=h.reshape(B*8,H).float()
    for e in range(10):
        gate=F.silu(flat@l.gate[e].T); up=flat@l.up[e].T
        _=(gate*up)@l.dn[e].T
    gv=torch.sigmoid(flat@l.sgw.T)
    so=(F.silu(flat@l.sg.T)*(flat@l.su.T))@l.sd.T
    h=(h+(gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)
    return h

def forward_opt(h, l):
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+1e-6)*l.n
    if hasattr(l,'o_t'):  # Full attention with O transposed
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
    # Stack all 10 experts → ONE batch H2D
    g=torch.stack(l.gu_cpu).float().to(DEV)
    d=torch.stack(l.dn_cpu).float().to(DEV)
    for e in range(10):
        gw,uw=g[e,:1024],g[e,1024:]
        _=(F.silu(flat@gw.T)*(flat@uw.T))@d[e].T
    gv=torch.sigmoid(flat@l.sgw.T)
    so=(F.silu(flat@l.sg.T)*(flat@l.su.T))@l.sd.T
    h=(h+(gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)
    return h

# ═══════════ Benchmark ═══════════
orig_F=OrigFull(); orig_D=OrigDelta()
opt_F=OptFull(); opt_D=OptDelta()
h_orig=torch.randn(B,8,H,dtype=torch.bfloat16,device=DEV)
h_opt=h_orig.clone()

# Original benchmark
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0=time.perf_counter()
for _ in range(N):
    for lid in range(60):
        h_orig=forward_orig(h_orig, orig_F if lid%4==3 else orig_D, lid%4==3)
torch.cuda.synchronize()
ms_orig=(time.perf_counter()-t0)/N*1000
vram_orig=torch.cuda.max_memory_allocated()/1024**3

# Optimized benchmark
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0=time.perf_counter()
for _ in range(N):
    for lid in range(60):
        h_opt=forward_opt(h_opt, opt_F if lid%4==3 else opt_D)
torch.cuda.synchronize()
ms_opt=(time.perf_counter()-t0)/N*1000
vram_opt=torch.cuda.max_memory_allocated()/1024**3

print(f'ORIGINAL:  {ms_orig:6.0f}ms/60L → {B*1000/(ms_orig/60):5.0f} tok/s | VRAM {vram_orig:.1f}GB')
print(f'OPTIMIZED: {ms_opt:6.0f}ms/60L → {B*1000/(ms_opt/60):5.0f} tok/s | VRAM {vram_opt:.1f}GB')
print(f'Speedup: {ms_orig/ms_opt:.1f}x | VRAM saved: {vram_orig-vram_opt:.1f}GB')

bench('ORIGINAL (current)', h_orig, orig_F, orig_D, forward_orig)
bench('OPTIMIZED (gate+O+torch)', h_opt, opt_F, opt_D, forward_opt)

print('\nSummary: all optimizations applied (virtual weights)')
