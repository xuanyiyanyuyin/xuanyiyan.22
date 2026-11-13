# 快测: torch SDPA 在 FLUX.2 单块参数(48 heads, head_dim=128)下, 序列长度到多少会非法访问
import torch, time
DEV = 'cuda'
B, HEADS, DIM_HEAD = 1, 48, 128
for S in [2560, 14592, 25600, 32768, 33280, 51712]:
    torch.cuda.synchronize()
    try:
        q = torch.randn(B, HEADS, S, DIM_HEAD, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(B, HEADS, S, DIM_HEAD, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(B, HEADS, S, DIM_HEAD, device=DEV, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        print(f'S={S} OK  ({time.perf_counter()-t0:.2f}s)  out={tuple(out.shape)}', flush=True)
        del q, k, v, out
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'S={S} FAIL: {type(e).__name__}: {str(e)[:120]}', flush=True)
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
print('SDPA_TEST_DONE', flush=True)
