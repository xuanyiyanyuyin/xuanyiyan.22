"""LTX-2.5 DiffVAE NABlock stub — 测 (3,7,7) 窗口注意力 + 2步Euler 耗时
用法: python ltx25_diffvae_stub.py
测: 在 2K (2560x1408x9) patchified 规模下, NABlock 窗口注意力的实际耗时
"""
import sys, time, torch, torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class WindowAttention3D(nn.Module):
    """(kt,kh,kw) 邻域注意力 — PyTorch 原生 (unfold/gather + mask), 不用 natten"""
    def __init__(self, dim, kernel_size=(3,7,7), head_dim=64):
        super().__init__()
        self.dim = dim
        self.kernel = kernel_size
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        # 预计算窗口偏移 (在 T,H,W 展平网格中)
        kt, kh, kw = kernel_size
        off = []
        for t in range(kt):
            for h in range(kh):
                for w in range(kw):
                    off.append((t-kt//2)*1 + (h-kh//2)*1 + (w-kw//2)*1)  # 占位, 下面重算
        # 正确偏移: 展平索引 = t*H*W + h*W + w
        self.offsets = torch.tensor(
            [(dt)*(1) + (dh)*(1) + (dw)*(1) for dt in range(kt) for dh in range(kh) for dw in range(kw)],
            dtype=torch.long)
        # 需要 H,W 来算实际偏移 — 用 forward 时动态计算

    def forward(self, x, H, W):
        """x: (B, T*H*W, dim). 返回 (B, T*H*W, dim). B=1, 分块处理保内存"""
        B, N, C = x.shape
        assert B == 1, "stub 只支持 B=1"
        T = N // (H * W)
        kt, kh, kw = self.kernel
        # 动态计算 3D 偏移: 展平索引偏移 = dt*H*W + dh*W + dw
        off = torch.empty(kt*kh*kw, dtype=torch.long, device=x.device)
        i = 0
        for dt in range(kt):
            for dh in range(kh):
                for dw in range(kw):
                    off[i] = dt*H*W + dh*W + dw
                    i += 1
        center = (kt//2)*H*W + (kh//2)*W + (kw//2)
        off = off - center  # 相对中心的偏移

        qkv = self.qkv(x)  # (B,N,3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim)
        k = k.view(B, N, self.n_heads, self.head_dim)
        v = v.view(B, N, self.n_heads, self.head_dim)
        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)

        # 分块邻域注意力 (B=1)
        out = torch.empty_like(x)
        CHUNK = 32768
        for start in range(0, N, CHUNK):
            end = min(start+CHUNK, N)
            ck = end - start
            base = torch.arange(start, end, device=x.device)[:, None]  # (ck,1)
            nidx_c = (base + off[None, :]).clamp(0, N-1)  # (ck,K)
            qc = q[0, start:end]   # (ck,NH,HD)
            kc = k[0][nidx_c]      # (ck,K,NH,HD)
            vc = v[0][nidx_c]
            attn = (qc[:, None, :, :] * kc).sum(-1)  # (ck,K,NH)
            attn = torch.softmax(attn, dim=1)
            oc = (attn[:, :, :, None] * vc).sum(1)   # (ck,NH,HD)
            out[0, start:end] = oc.reshape(ck, C)
        out = self.proj(out)
        return out


class DiffNABlock(nn.Module):
    """diffusion NA block: AdaLN-Zero + 窗口注意力 + FFN"""
    def __init__(self, dim=128, kernel=(3,7,7), head_dim=64):
        super().__init__()
        self.dim = dim
        self.ada = nn.Linear(384, dim*4, bias=True)  # shift/scale for attn + ffn
        self.attn = WindowAttention3D(dim, kernel, head_dim)
        self.norm1 = nn.RMSNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x, t_emb, H, W):
        """AdaLN-Zero 简化: 从 t_emb 生成 scale/shift"""
        ada = self.ada(t_emb)  # (B, 6*dim)
        s1, b1, s2, b2 = ada.chunk(4, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None, :]) + b1[:, None, :]
        h = h + self.attn(h, H, W)
        h = self.norm2(h) * (1 + s2[:, None, :]) + b2[:, None, :]
        h = h + self.ff(h)
        return h


def run_stub(T, H, W, B=1, C=128, depth=8, nsteps=2, chunk=None):
    """在 T*H*W token 网格上跑 depth 个 DiffNABlock, nsteps 步"""
    N = T*H*W
    x = torch.randn(B, N, C, device=DEV, dtype=torch.bfloat16)
    t_emb = torch.randn(B, 384, device=DEV, dtype=torch.bfloat16)
    blocks = [DiffNABlock(C, (3,7,7), 64).to(torch.bfloat16).to(DEV) for _ in range(depth)]

    # warmup
    with torch.inference_mode():
        for blk in blocks[:2]:
            blk(x, t_emb, H, W)

    torch.cuda.synchronize() if DEV=='cuda' else None
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(nsteps):
            cur = x
            for blk in blocks:
                cur = blk(cur, t_emb, H, W)
            # Euler step 简化 (v model)
            cur = cur - cur  # 占位
    torch.cuda.synchronize() if DEV=='cuda' else None
    dt = time.perf_counter() - t0
    return dt, N


if __name__ == '__main__':
    print(f'设备: {DEV} (bf16)')
    print(f"{'规模':>18} {'tokens':>10} {'2步耗时':>10} {'每token':>10}")
    scales = [
        (9, 80, 44),     # latent上采样1档
        (9, 160, 88),    # 中档
        (9, 320, 176),   # 中高档
        (9, 640, 352),   # 2K patchified (2560/4 x 1408/4)
    ]
    for T,H,W in scales:
        try:
            dt, N = run_stub(T, H, W)
            print(f'{T}x{H}x{W:>12} {N:>10,} {dt*1000:>9.0f}ms {dt/N*1e6:>9.1f}us/tok')
        except Exception as e:
            print(f'{T}x{H}x{W:>12} 失败: {str(e)[:60]}')
