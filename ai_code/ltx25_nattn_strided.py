"""as_strided 3D 窗口注意力 — 零拷贝提取窗口 + 分块 matmul (替换 naive gather)
用法: python ltx25_nattn_strided.py T H W [C]
输入: (B,T,H,W,C) → 输出 (B,T,H,W,C)
按千文方案: F.pad → as_strided 窗口 view → 分块 Q@K^T
"""
import sys, time, torch, torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def window_attn_strided(q, k, v, kernel=(3,7,7)):
    """q,k,v: (B,T,H,W,NH,HD). 返回 (B,T,H,W,NH*HD)"""
    B, T, H, W, NH, HD = q.shape
    kt, kh, kw = kernel
    N = T * H * W
    # 1. pad (T,H,W) 到 kernel 边界: permute 让 (T,H,W) 在最后, pad, 再转回
    def _pad(x):
        x = x.permute(0, 4, 5, 1, 2, 3)      # (B,NH,HD,T,H,W)
        x = F.pad(x, (kw//2, kw//2, kh//2, kh//2, kt//2, kt//2))  # pad W,H,T
        return x.permute(0, 3, 4, 5, 1, 2)   # 回 (B,T,H,W,NH,HD)
    qp = _pad(q); kp = _pad(k); vp = _pad(v)
    # 2. as_strided 提取所有窗口 (零拷贝)
    sB, sT, sH, sW, sNH, sHD = qp.stride()
    wsize = (B, T, H, W, kt, kh, kw, NH, HD)
    wstride = (sB, sT, sH, sW, sT, sH, sW, sNH, sHD)
    kw_ = kp.as_strided(wsize, wstride)  # (B,T,H,W,kt,kh,kw,NH,HD)
    vw_ = vp.as_strided(wsize, wstride)
    # 展平窗口 → (B, N, K, NH, HD)
    kw_ = kw_.reshape(B, N, kt*kh*kw, NH, HD)
    vw_ = vw_.reshape(B, N, kt*kh*kw, NH, HD)
    # center token = q (B,N,NH,HD)
    qc = q.reshape(B, N, NH, HD)
    # 3. 分块 matmul
    out = torch.empty_like(qc)
    CHUNK = 16384
    for s in range(0, N, CHUNK):
        e = min(s+CHUNK, N)
        ck = e - s
        q_chunk = qc[:, s:e]              # (B,ck,NH,HD)
        k_chunk = kw_[:, s:e]             # (B,ck,K,NH,HD)
        v_chunk = vw_[:, s:e]
        attn = (q_chunk.unsqueeze(2) * k_chunk).sum(-1)  # (B,ck,K,NH)
        attn = torch.softmax(attn, dim=2)
        out[:, s:e] = (attn.unsqueeze(-1) * v_chunk).sum(2)  # (B,ck,NH,HD)
    return out.reshape(B, T, H, W, NH*HD)


def test(T, H, W, C=256, head_dim=64):
    B = 1
    NH = C // head_dim
    q = torch.randn(B, T, H, W, NH, head_dim, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, W, NH, head_dim, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, W, NH, head_dim, device=DEV, dtype=torch.bfloat16)
    # warmup
    window_attn_strided(q, k, v)
    torch.cuda.synchronize() if DEV=='cuda' else None
    t0 = time.perf_counter()
    out = window_attn_strided(q, k, v)
    torch.cuda.synchronize() if DEV=='cuda' else None
    dt = time.perf_counter() - t0
    N = T*H*W
    peak = torch.cuda.max_memory_allocated()/2**30 if DEV=='cuda' else 0
    print(f'{T}x{H}x{W}: tokens={N:,} 耗时={dt*1000:.0f}ms  输出{tuple(out.shape)}  峰值显存={peak:.1f}GB')

if __name__ == '__main__':
    args = [int(x) for x in sys.argv[1:]]
    if len(args) >= 3:
        test(*args[:3], args[3] if len(args)>3 else 256)
    else:
        print('按千文放大纪律: 小→中→大, 每级看显存')
        for T,H,W in [(3,32,16), (3,88,44), (9,352,640)]:
            test(T,H,W)
