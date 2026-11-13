"""
40-batch H2D 优化测试 · 虚拟权重 (内存优化版)
复用权重对象，对比 10/20/40 专家堆叠传输
"""
import torch, torch.nn.functional as F, time
DEV='cuda'; H=4096; B=1; N=5

def make_opt_full(n_exp):
    class O:
        def __init__(self):
            self.n=torch.randn(H,device=DEV)
            self.q=torch.randn(16384,H,device=DEV); self.k=torch.randn(512,H,device=DEV)
            self.v=torch.randn(512,H,device=DEV); self.o_t=torch.randn(8192,4096,device=DEV)
            self.gu_cpu=[torch.randn(2048,H,dtype=torch.bfloat16) for _ in range(n_exp)]
            self.dn_cpu=[torch.randn(H,1024,dtype=torch.bfloat16) for _ in range(n_exp)]
            self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
            self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
            self.sd=torch.randn(H,1024,device=DEV)
    return O

def make_opt_delta(n_exp):
    class O:
        def __init__(self):
            self.n=torch.randn(H,device=DEV)
            self.w1=torch.randn(H,H,device=DEV); self.w2=torch.randn(H,H,device=DEV)
            self.gu_cpu=[torch.randn(2048,H,dtype=torch.bfloat16) for _ in range(n_exp)]
            self.dn_cpu=[torch.randn(H,1024,dtype=torch.bfloat16) for _ in range(n_exp)]
            self.rw=torch.randn(512,H,device=DEV); self.sgw=torch.randn(1,H,device=DEV)
            self.sg=torch.randn(1024,H,device=DEV); self.su=torch.randn(1024,H,device=DEV)
            self.sd=torch.randn(H,1024,device=DEV)
    return O

def forward_opt(h, l, n_exp):
    hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+1e-6)*l.n
    if hasattr(l,'o_t'):
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
    g=torch.stack(l.gu_cpu).float().to(DEV)
    d=torch.stack(l.dn_cpu).float().to(DEV)
    for e in range(n_exp):
        gw,uw=g[e,:1024],g[e,1024:]
        _=(F.silu(flat@gw.T)*(flat@uw.T))@d[e].T
    gv=torch.sigmoid(flat@l.sgw.T)
    so=(F.silu(flat@l.sg.T)*(flat@l.su.T))@l.sd.T
    h=(h+(gv*so).to(torch.bfloat16).reshape_as(h)).to(torch.bfloat16)
    del g,d; return h

for n_exp, label in [(10,"10专家"), (20,"20专家"), (40,"40专家")]:
    OF=make_opt_full(n_exp); OD=make_opt_delta(n_exp)
    # Reuse same 2 weight objects for all 60 layers!
    of=OF(); od=OD()
    h=torch.randn(B,8,H,dtype=torch.bfloat16,device=DEV)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    # Warmup
    for lid in range(2): h=forward_opt(h, of if lid%2==0 else od, n_exp)
    torch.cuda.synchronize()

    t0=time.perf_counter()
    for _ in range(N):
        for lid in range(60):
            h=forward_opt(h, of if lid%4==3 else od, n_exp)
    torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)/N*1000
    vram=torch.cuda.max_memory_allocated()/1024**3

    tok_s=60*B*1000/ms; s_tok=ms/(60*B)/1000
    print(f'{label:8s} | {ms:6.0f}ms/60层 | {tok_s:5.1f} tok/s | {s_tok:5.2f} s/token | VRAM {vram:.1f}GB')
    del of,od,OF,OD,h
