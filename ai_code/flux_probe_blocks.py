"""FLUX 显存容量探针: 加载后逐块 .to(cuda), 看能常驻多少块不OOM.
用法: python flux_probe_blocks.py [w] [h] [extra_gb]
"""
import torch, time, sys, os
sys.path.insert(0, r'D:\ai')
MODEL = r'D:\ai\models\FLUX.1-dev'
DEV = 'cuda'
TOTAL_GB = 12.0

def gpu_used():
    # 该进程已分配(含缓存)
    return torch.cuda.memory_allocated()/1024**3

def main():
    t0 = time.perf_counter()
    from diffusers import FluxImg2ImgPipeline
    print('加载 pipeline (CPU)...', flush=True)
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = pipe.to('cpu')
    torch.cuda.empty_cache()
    t = pipe.transformer
    nb_d = len(t.transformer_blocks)
    nb_s = len(t.single_transformer_blocks)
    print(f'加载 {time.perf_counter()-t0:.0f}s | 双块{nb_d} 单块{nb_s}', flush=True)

    # 单块权重大小
    bd = sum(p.numel()*p.element_size() for p in t.transformer_blocks[0].parameters())/1024**3
    bs = sum(p.numel()*p.element_size() for p in t.single_transformer_blocks[0].parameters())/1024**3
    print(f'双块~{bd:.3f}GB  单块~{bs:.3f}GB', flush=True)

    # setup 模块
    for m in (t.x_embedder, t.time_text_embed, t.context_embedder, t.pos_embed, t.norm_out, t.proj_out):
        m.to(DEV)
    torch.cuda.empty_cache()
    base = gpu_used()
    print(f'setup 上GPU后 used={base:.2f}GB', flush=True)

    # 激活余量: 至少留 2.5GB 给 384 输入激活 + VAE临时
    budget = TOTAL_GB - 2.5
    print(f'给块权重预算: {budget:.2f}GB (12GB总 - 2.5GB激活余量 - {base:.2f}setup)', flush=True)

    # 先放双块
    print('\n--- 逐块放双块 (每放一块打印) ---', flush=True)
    cnt_d = 0
    for i in range(nb_d):
        t.transformer_blocks[i].to(DEV)
        used = gpu_used()
        cnt_d += 1
        if i % 2 == 1 or i == nb_d-1:
            print(f'  放完第{i+1}个双块: used={used:.2f}GB  累计{cnt_d}双块', flush=True)
        if used > budget:
            print(f'  [!] 第{i+1}个双块后 used={used:.2f}GB 超预算, 回退', flush=True)
            t.transformer_blocks[i].to('cpu')
            cnt_d -= 1
            break

    # 再放单块
    print('\n--- 继续放单块 (每放一块打印) ---', flush=True)
    cnt_s = 0
    for j in range(nb_s):
        t.single_transformer_blocks[j].to(DEV)
        used = gpu_used()
        cnt_s += 1
        if j % 5 == 4 or j == nb_s-1:
            print(f'  放完第{j+1}个单块: used={used:.2f}GB  累计{cnt_s}单块', flush=True)
        if used > budget:
            print(f'  [!] 第{j+1}个单块后 used={used:.2f}GB 超预算, 回退', flush=True)
            t.single_transformer_blocks[j].to('cpu')
            cnt_s -= 1
            break

    print(f'\n✅ 结论: 显存可常驻 {cnt_d} 双块 + {cnt_s} 单块 = {cnt_d+cnt_s} 块 (约{cnt_d*bd+cnt_s*bs:.2f}GB)', flush=True)
    print(f'   (给激活留了 {TOTAL_GB-2.5-(cnt_d*bd+cnt_s*bs+base):.2f}GB 余量)', flush=True)

if __name__ == '__main__':
    main()
