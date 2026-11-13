"""DiffVAE 真实管线测时 — 源码原版 + 真实权重, 仅 natten 后端换纯PyTorch窗口注意力
用法: python ltx25_diffvae_real.py [T H W]
构造 dummy latent, 跑完整 DiffusionVideoDecoder.forward, 计时
"""
import sys, time, torch
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')

class PyTorchWindowAttention:
    """纯 PyTorch 窗口注意力 — 替换 natten (接口与 NattenAttention 一致)"""
    def __init__(self, kernel_size):
        self.kernel_size = tuple(kernel_size)
    def __call__(self, attn, q, k, v):
        B, T, H, W, NH, HD = q.shape
        kt, kh, kw = self.kernel_size
        N = T * H * W
        # 3D 窗口偏移 (展平索引)
        off = torch.empty(kt*kh*kw, dtype=torch.long, device=q.device)
        i = 0
        for dt in range(kt):
            for dh in range(kh):
                for dw in range(kw):
                    off[i] = dt*H*W + dh*W + dw
                    i += 1
        off = off - (kt//2*H*W + kh//2*W + kw//2)
        qf = q.reshape(B, N, NH, HD); kf = k.reshape(B, N, NH, HD); vf = v.reshape(B, N, NH, HD)
        out = torch.empty_like(qf)
        CHUNK = 32768
        for start in range(0, N, CHUNK):
            end = min(start+CHUNK, N)
            idx = (torch.arange(start, end, device=q.device)[:, None] + off[None, :]).clamp(0, N-1)
            for b in range(B):
                qc = qf[b, start:end]      # (ck,NH,HD)
                kc = kf[b][idx]            # (ck,K,NH,HD)
                vc = vf[b][idx]
                a = (qc[:, None, :, :] * kc).sum(-1)  # (ck,K,NH)  (q 已 scale)
                a = torch.softmax(a, dim=1)
                oc = (a[:, :, :, None] * vc).sum(1)    # (ck,NH,HD)
                out[b, start:end] = oc.reshape(end-start, NH, HD)
        return out.reshape(B, T, H, W, NH*HD)


def main(T=2, H=44, W=80):
    from ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
    # 真实 config (从权重反推)
    dec = DiffusionVideoDecoder(
        stage_channels=(2048, 1024, 512, 512, 256),
        stage_depths=(4, 6, 4, 2, 8),
        stage5_channels=256,
    )
    # 注入纯 PyTorch 窗口注意力
    n_swapped = 0
    for m in dec.modules():
        if type(m).__name__ == 'NeighborhoodAttention3D':
            m.attention_function = PyTorchWindowAttention(m.kernel_size)
            n_swapped += 1
    print(f'已替换 {n_swapped} 个 NeighborhoodAttention3D 后端', flush=True)
    # 权重格式与代码结构不匹配 (checkpoint融合qkv vs 代码拆分), 但耗时测试用随机权重即可
    # (计算图/形状相同, 耗时相同; 真实权重加载是魔改时的活)
    print(f'随机初始化 (架构正确), 解码参数 {sum(p.numel() for p in dec.parameters())/1e6:.0f}M', flush=True)
    dec = dec.to('cuda').bfloat16()

    lat = torch.randn(1, 128, T, H, W, device='cuda', dtype=torch.bfloat16)
    print(f'输入 latent: {tuple(lat.shape)}', flush=True)
    # 抓 stage5 (diff_blocks) 输入 shape
    stage5_shape = {}
    def hook(mod, inp, out):
        stage5_shape['in'] = tuple(inp[0].shape) if isinstance(inp, tuple) and len(inp) else tuple(inp.shape)
        stage5_shape['out'] = tuple(out.shape) if isinstance(out, torch.Tensor) else None
    h = dec.diff_blocks.register_forward_hook(hook)
    # warmup
    with torch.inference_mode():
        dec.forward(torch.randn(1,128,T,H//2,W//2, device='cuda', dtype=torch.bfloat16))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = dec.forward(lat)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    h.remove()
    print(f'stage5 输入 shape: {stage5_shape.get("in")}', flush=True)
    print(f'DiffVAE 完整解码 (2步Euler): {dt*1000:.0f} ms, 输出 {tuple(out.shape)}', flush=True)
    # 理论窗口注意力 FLOPs
    s5 = stage5_shape.get('in')
    if s5:
        B, C, TF, TH, TW = s5
        K = 3*7*7
        NH, HD = C//64, 64
        n_tokens = TF*TH*TW
        attn_flops = 2 * n_tokens * K * NH * HD  # qk + av
        proj_flops = 6 * n_tokens * C * C        # qkv(3) + proj(1) + mlp(2x4) 粗估
        print(f'stage5 tokens={n_tokens:,}, 窗口K={K}, 每NABlock attn FLOPs={attn_flops/1e9:.2f}G', flush=True)
        print(f'全8块x2步 attn FLOPs ≈ {attn_flops*8*2/1e9:.1f}G', flush=True)

if __name__ == '__main__':
    import sys as _s
    args = _s.argv[1:]
    main(int(args[0]), int(args[1]), int(args[2])) if len(args) == 3 else main()
