"""FLUX 显存诊断 step2: 单块真实forward, 每步前后看 allocated 涨落.
用法: python flux_diag.py <img> [w] [h] [n_steps]
"""
import torch, time, sys, os
sys.path.insert(0, r'D:\ai')
MODEL = r'D:\ai\models\FLUX.1-dev'
DEV = 'cuda'

def mem(tag):
    a = torch.cuda.memory_allocated()/1024**3
    r = torch.cuda.memory_reserved()/1024**3
    free,_ = torch.cuda.mem_get_info()
    print(f'  [{tag:34s}] alloc={a:6.2f}GB reserved={r:6.2f}GB free={free/1024**3:6.2f}GB', flush=True)

def main():
    W = int(sys.argv[1]) if len(sys.argv)>1 else 384
    H = int(sys.argv[2]) if len(sys.argv)>2 else 384
    NSTEP = int(sys.argv[3]) if len(sys.argv)>3 else 6

    t0 = time.perf_counter()
    from diffusers import FluxImg2ImgPipeline
    print('加载 pipeline...', flush=True)
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = pipe.to('cpu')
    torch.cuda.empty_cache()
    t = pipe.transformer
    blk = t.transformer_blocks[0]
    nparam = sum(p.numel()*p.element_size() for p in blk.parameters())
    print(f'加载 {time.perf_counter()-t0:.0f}s | 一个双块权重={nparam/1024**3:.3f}GB', flush=True)
    mem('after load')

    blk = blk.to(DEV)
    torch.cuda.empty_cache()
    mem('after block0 on GPU')

    # 构造输入: block 期望的是已过 x_embedder 的 3072 维 hidden_states
    seq_img = (H//16) * (W//16)     # 384/16=24 → 576
    seq_txt = 512
    hidden = 3072
    lat  = torch.randn(1, seq_img, hidden, device=DEV, dtype=torch.bfloat16)
    enc  = torch.randn(1, seq_txt, hidden, device=DEV, dtype=torch.bfloat16)
    temb = torch.randn(1, hidden, device=DEV, dtype=torch.bfloat16)
    rot  = (torch.randn(seq_img+seq_txt, 128, device=DEV, dtype=torch.bfloat16),
            torch.randn(seq_img+seq_txt, 128, device=DEV, dtype=torch.bfloat16))  # (cos,sin) 对
    mem('after input alloc')

    # 单块 forward NSTEP 次, 模拟真实循环: 每次 del 旧引用再换新, 看 alloc 是否累加(不主动empty_cache)
    h, e = lat, enc
    print(f'\n--- 真实循环 [no_grad下]: 每步 del旧+换新 不 empty_cache, 看alloc是否还涨 ---', flush=True)
    for i in range(NSTEP):
        with torch.no_grad():    # 关键: 推理必须 no_grad, 否则 autograd graph 累积
            new_h, new_e = blk(hidden_states=h, encoder_hidden_states=e, temb=temb, image_rotary_emb=rot)
        del h, e                 # 显式释放旧引用
        h, e = new_h, new_e
        torch.cuda.synchronize()
        mem(f'after step {i+1} (no_grad)')

    print('\n--- 关键: 仅 del 引用(不empty_cache), alloc 是否回落 ---', flush=True)
    # 这里模拟"整层跑完": 只留下 block0 权重, 释放所有激活
    del h, e
    # 注意不调用 empty_cache, 看 allocator 是否因为 del 自动还
    torch.cuda.synchronize()
    mem('after del (NO empty_cache)')
    print('\n诊断结论: 若[after del]回落→del即释放; 若仍高→必须empty_cache或缓存复用', flush=True)

if __name__ == '__main__':
    main()
