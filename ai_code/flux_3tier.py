"""
FLUX.1-dev 三层搬运 img2img —— 照 inference_stream.py "GPU只算不存":
权重全程只在 RAM 一份; 显存只驻留当前正在算的 1 个块(用预分配共享槽 copy_ 载入)。
每块算完: block 输出 + 该块临时激活立即无引用 → 显存空间自然被下一块复用, 不累积、不挤共享。
不常驻多块、不 empty_cache 玄学、权重不双份。
"""
import torch, time, os, sys, math
import numpy as np
sys.path.insert(0, r'D:\ai')
# ══ [2026-10-02] 内存卫生：mmap 进来的权重页是干净的，用完/退出时归还给系统 ══
#    自带 sys.path 兜底 —— 放哪都不会 ImportError（2026-10-02 踩过：块被插到
#    sys.path.insert 之前，导致 flux2_encode_gen 一加载就 ModuleNotFoundError）
try:
    import sys as _mh_sys
    if r'D:\ai' not in _mh_sys.path:
        _mh_sys.path.insert(0, r'D:\ai')
    import mem_hygiene as _mh
    _mh.trim('start')
    import atexit as _atexit
    _atexit.register(lambda: _mh.trim('exit'))
except Exception:
    _mh = None

MODEL = r'D:\ai\models\FLUX.1-dev'
OUT   = r'D:\ai\output\images'
DEV   = 'cuda'

def _patch_gpu_only(t, resident_d=0):
    """替换 forward 为三层流式 + 常驻:
    - 前 resident_d 个双块: 一次性 .to(DEV) 常驻, 每步不搬(省H2D)
    - 其余双块 + 全部单块: 预分配共享GPU槽流式, 每块copy权重→算→还原
    no_grad 包住整个推理 → 无 autograd 泄漏. 显存恒定 = 常驻块 + 1套槽 + 激活.
    """
    nb_d = len(t.transformer_blocks)
    nb_s = len(t.single_transformer_blocks)
    resident_d = max(0, min(resident_d, nb_d))   # 双块常驻数
    # setup 模块常驻 GPU (小)
    for m in (t.x_embedder, t.time_text_embed, t.context_embedder, t.pos_embed, t.norm_out, t.proj_out):
        m.to(DEV)
    # 常驻双块一次性上 GPU
    for i in range(resident_d):
        t.transformer_blocks[i].to(DEV)
    # 每类非驻留块各预分配 1 套共享 GPU 槽
    def _slots(blk):
        s = {}
        with torch.no_grad():
            for name, p in blk.named_parameters():
                s[name] = torch.empty_like(p, device=DEV)
        return s
    slots_d = _slots(t.transformer_blocks[resident_d]) if resident_d < nb_d else {}
    slots_s = _slots(t.single_transformer_blocks[0])
    torch.cuda.empty_cache()

    # 保存 CPU 原始参数备份 (非驻留块用, 权重只在RAM一份)
    _cpu_backup = {}
    def _backup(blk):
        for name, p in blk.named_parameters():
            _cpu_backup[id(p)] = p.data  # CPU 张量
    def _restore(blk):
        for name, p in blk.named_parameters():
            p.data = _cpu_backup.pop(id(p), p.data)

    def forward(hidden_states, timestep, guidance=None, pooled_projections=None,
                encoder_hidden_states=None, txt_ids=None, img_ids=None,
                joint_attention_kwargs=None, return_dict=True, **kw):
        with torch.no_grad():   # 关键: no_grad, 否则 autograd graph 累积泄漏 (照 inference_stream)
            return _forward_nograd(hidden_states, timestep, guidance, pooled_projections,
                                   encoder_hidden_states, txt_ids, img_ids, joint_attention_kwargs)

    def _forward_nograd(hidden_states, timestep, guidance, pooled_projections,
                        encoder_hidden_states, txt_ids, img_ids, joint_attention_kwargs):
        hidden_states = t.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (t.time_text_embed(timestep, pooled_projections)
                if guidance is None else t.time_text_embed(timestep, guidance, pooled_projections))
        enc = t.context_embedder(encoder_hidden_states)
        if txt_ids is not None and txt_ids.ndim == 3: txt_ids = txt_ids[0]
        if img_ids is not None and img_ids.ndim == 3: img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        rotary = t.pos_embed(ids)
        jkw = dict(joint_attention_kwargs or {})

        def _run_stream(block, slots):
            """流式块: RAM权重→共享槽→算→还原. 返回 (enc_out, hidden_out)"""
            _backup(block)
            for name, p in block.named_parameters():
                slots[name].copy_(p.data, non_blocking=False)
                p.data = slots[name]
            new_out = block(hidden_states=hidden_states, encoder_hidden_states=enc, temb=temb,
                            image_rotary_emb=rotary, joint_attention_kwargs=jkw)
            _restore(block)
            return new_out

        def _run_resident(block):
            """常驻块: 权重已在GPU, 直接算"""
            return block(hidden_states=hidden_states, encoder_hidden_states=enc, temb=temb,
                         image_rotary_emb=rotary, joint_attention_kwargs=jkw)

        for i in range(nb_d):
            if i < resident_d:
                new_out = _run_resident(t.transformer_blocks[i])
            else:
                new_out = _run_stream(t.transformer_blocks[i], slots_d)
            del enc, hidden_states
            enc, hidden_states = new_out
            del new_out
        for j in range(nb_s):
            new_out = _run_stream(t.single_transformer_blocks[j], slots_s)
            del enc, hidden_states
            enc, hidden_states = new_out
            del new_out
        del enc

        hidden_states = t.norm_out(hidden_states, temb)
        output = t.proj_out(hidden_states)
        return (output,)

    t.forward = forward
    return resident_d

def _align_wh(w, h, mult=64):
    """把宽高对齐到 mult 的整数倍 (FLUX VAE 8x下采样 + pack 需2整除 → 像素须被16整除.
    对齐到64更稳, 与你文生图2560x1408一致). 返回 (w,h)."""
    w = max(mult, round(w / mult) * mult)
    h = max(mult, round(h / mult) * mult)
    return int(w), int(h)

def _auto_resident(w, h, resident_d):
    """按分辨率自动选常驻块数. resident_d=0 表示自动.
    依据实测: 12GB 下 常驻块(0.633GB/块) + 激活(随token涨) 必须 < 12GB.
      token = (w/16)*(h/16). 激活预算约 = 12 - 常驻权重.
    经验表: 激活随 token² 涨. 用 token 估算激活, 反推可常驻块数."""
    tok = (w // 16) * (h // 16)          # pack后latent token数
    # 实测锚点: 768x1376(tok4128)→激活~2GB→常驻12; 2560x1408(tok14080)→激活~6GB→常驻6
    # 用线性近似: 激活GB ≈ 1.5 + tok/2500
    act_gb = 1.5 + tok / 2500.0
    per_block = 0.633
    budget = 12.0 - act_gb - 0.6         # 总 - 激活 - setup/流式槽
    autoK = max(1, min(19, int(budget // per_block)))
    if resident_d == 0:
        return autoK
    return resident_d

def img2img_3tier(prompt, init_image, strength=0.35, steps=28, width=768, height=1376,
                  guidance_scale=7.0, max_seq=512, out_path=None, resident_d=12):
    """FLUX 三层搬运 img2img.
    参数指导:
      - strength 0-1 (低0.30~0.40修细节保结构, 高0.6+大改)
      - guidance_scale 7.0 官方img2img默认; 文生图若用3.5, 图会偏灰
      - steps 去噪总步数(受strength截断, 实际步=steps*strength); 28=均衡 50=更细
      - width/height 会被自动对齐到64倍数(否则VAE打包尺寸不合法)
      - resident_d 常驻双块数(0~14; 12GB实测14块8.98GB为上限, 12稳妥提速~3x)
    """
    width, height = _align_wh(width, height)
    # 大图自动降常驻 (resident_d=0 自动), 给激活让路防OOM
    use_rd = _auto_resident(width, height, resident_d)
    from PIL import Image
    from diffusers import FluxImg2ImgPipeline
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    if isinstance(init_image, str):
        init_image = Image.open(init_image).convert('RGB')
    t0 = time.perf_counter()
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = pipe.to('cpu')
    r = _patch_gpu_only(pipe.transformer, resident_d=use_rd)
    print(f'[3tier] pipe loaded, 常驻{r}双块 + 其余流式 (auto_rd={use_rd})', flush=True)

    # 文本编码 CPU 跑, 结果挪GPU
    pipe.text_encoder.to('cpu'); pipe.text_encoder_2.to('cpu')
    (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
        prompt=prompt, prompt_embeds=None, pooled_prompt_embeds=None,
        device='cpu', num_images_per_prompt=1, max_sequence_length=max_seq, lora_scale=None)
    prompt_embeds = prompt_embeds.to(DEV); pooled_prompt_embeds = pooled_prompt_embeds.to(DEV)
    text_ids = text_ids.to(DEV)
    torch.cuda.empty_cache()
    print(f'[3tier] prompt encoded {time.perf_counter()-t0:.0f}s', flush=True)

    proc = pipe.image_processor
    pipe.vae.to(DEV)
    with torch.no_grad():
        init = proc.preprocess(init_image, height=height, width=width).to(DEV, dtype=torch.bfloat16)
        vae_lat = pipe.vae.encode(init).latent_dist.sample()
        vae_lat = (vae_lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    pipe.vae.to('cpu')
    nch = pipe.transformer.config.in_channels // 4
    H_pack = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    W_pack = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    lat_ids = pipe._prepare_latent_image_ids(1, H_pack // 2, W_pack // 2, DEV, torch.bfloat16)
    print(f'[3tier] vae encoded {time.perf_counter()-t0:.0f}s', flush=True)

    image_seq_len = (H_pack // 2) * (W_pack // 2)
    mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    ts, _ = retrieve_timesteps(pipe.scheduler, steps, DEV, sigmas=sigmas, mu=mu)
    init_timestep = min(steps * strength, steps)
    t_start = int(max(steps - init_timestep, 0))
    ts = ts[t_start:]
    pipe.scheduler.set_begin_index(t_start)

    noise = torch.randn_like(vae_lat)
    vae_lat = pipe.scheduler.scale_noise(vae_lat, ts[:1], noise)
    lat = pipe._pack_latents(vae_lat, 1, nch, H_pack, W_pack).to(DEV)
    guidance = torch.full([1], guidance_scale, device=DEV, dtype=torch.float32).expand(lat.shape[0])
    torch.cuda.empty_cache()

    for i, t in enumerate(ts):
        timestep = t.expand(lat.shape[0]).to(lat.dtype)
        out = pipe.transformer(hidden_states=lat, timestep=timestep / 1000, guidance=guidance,
                               pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                               txt_ids=text_ids, img_ids=lat_ids, joint_attention_kwargs={},
                               return_dict=False)
        new_lat = pipe.scheduler.step(out[0], t, lat, return_dict=False)[0].to(lat.dtype)
        del out, lat          # 释放旧 lat + out, 防残留占显存
        lat = new_lat
        del new_lat
        torch.cuda.empty_cache()
        print(f'[3tier] step {i+1}/{len(ts)} {time.perf_counter()-t0:.0f}s', flush=True)

    # 解码前把 transformer 挪回 CPU, 释放全部权重显存给 VAE 解码 (大图解码需显存)
    pipe.transformer.to('cpu')
    for m in (pipe.transformer.x_embedder, pipe.transformer.time_text_embed, pipe.transformer.context_embedder,
              pipe.transformer.pos_embed, pipe.transformer.norm_out, pipe.transformer.proj_out):
        m.to('cpu')
    torch.cuda.empty_cache()
    pipe.vae.to(DEV)
    pipe.vae.enable_tiling()   # 4K解码峰值18.96G→2.6G, 12G必开
    with torch.no_grad():
        lat_cpu = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        del lat
        lat_u = pipe._unpack_latents(lat_cpu, height, width, pipe.vae_scale_factor)
        del lat_cpu
        img = pipe.vae.decode(lat_u, return_dict=False)[0]
        del lat_u
    img = pipe.image_processor.postprocess(img, output_type='pil')[0]
    torch.cuda.empty_cache()
    pipe.vae.to('cpu')
    fn = out_path or f'{OUT}/tier_{int(time.time())}.png'
    img.save(fn)
    print(f'[3tier] done {time.perf_counter()-t0:.0f}s -> {fn} ({img.size})', flush=True)
    return fn

def text2img_3tier(prompt, steps=50, width=2560, height=1408, guidance_scale=7.0,
                   max_seq=512, out_path=None, resident_d=0, seed=None):
    """文生图(三层搬运): 从随机噪声去噪, 不走VAE编码底图, 全部步数.
    同款 no_grad 三层搬运, 快且不泄漏."""
    width, height = _align_wh(width, height)
    use_rd = _auto_resident(width, height, resident_d)
    import torch
    from diffusers import FluxImg2ImgPipeline
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    t0 = time.perf_counter()
    pipe = FluxImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe = pipe.to('cpu')
    r = _patch_gpu_only(pipe.transformer, resident_d=use_rd)
    print(f'[t2i3tier] pipe loaded, 常驻{r}双块 (auto_rd={use_rd})', flush=True)

    # 文本编码 CPU, 结果挪GPU
    pipe.text_encoder.to('cpu'); pipe.text_encoder_2.to('cpu')
    (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
        prompt=prompt, prompt_embeds=None, pooled_prompt_embeds=None,
        device='cpu', num_images_per_prompt=1, max_sequence_length=max_seq, lora_scale=None)
    prompt_embeds = prompt_embeds.to(DEV); pooled_prompt_embeds = pooled_prompt_embeds.to(DEV)
    text_ids = text_ids.to(DEV)
    torch.cuda.empty_cache()
    print(f'[t2i3tier] prompt encoded {time.perf_counter()-t0:.0f}s', flush=True)

    nch = pipe.transformer.config.in_channels // 4
    H_pack = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    W_pack = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    lat_ids = pipe._prepare_latent_image_ids(1, H_pack // 2, W_pack // 2, DEV, torch.bfloat16)

    image_seq_len = (H_pack // 2) * (W_pack // 2)
    mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    ts, _ = retrieve_timesteps(pipe.scheduler, steps, DEV, sigmas=sigmas, mu=mu)
    pipe.scheduler.set_begin_index(0)

    # 文生图: 全随机噪声起 (不截断步数)
    g = torch.Generator(device=DEV).manual_seed(seed if seed is not None else int(time.time()) % 10000)
    shape = (1, nch, H_pack, W_pack)
    lat = torch.randn(shape, generator=g, device=DEV, dtype=torch.bfloat16)
    lat = pipe._pack_latents(lat, 1, nch, H_pack, W_pack).to(DEV)
    guidance = torch.full([1], guidance_scale, device=DEV, dtype=torch.float32).expand(lat.shape[0])
    torch.cuda.empty_cache()

    for i, t in enumerate(ts):
        timestep = t.expand(lat.shape[0]).to(lat.dtype)
        out = pipe.transformer(hidden_states=lat, timestep=timestep / 1000, guidance=guidance,
                               pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                               txt_ids=text_ids, img_ids=lat_ids, joint_attention_kwargs={},
                               return_dict=False)
        new_lat = pipe.scheduler.step(out[0], t, lat, return_dict=False)[0].to(lat.dtype)
        del out, lat
        lat = new_lat
        del new_lat
        torch.cuda.empty_cache()
        if i % 10 == 0 or i == len(ts) - 1:
            print(f'[t2i3tier] step {i+1}/{len(ts)} {time.perf_counter()-t0:.0f}s', flush=True)

    # 解码 (GPU解码, 快; 权重已释放, 逐段del降峰值防冲共享显存)
    pipe.transformer.to('cpu')
    for m in (pipe.transformer.x_embedder, pipe.transformer.time_text_embed, pipe.transformer.context_embedder,
              pipe.transformer.pos_embed, pipe.transformer.norm_out, pipe.transformer.proj_out):
        m.to('cpu')
    torch.cuda.empty_cache()
    pipe.vae.to(DEV)
    pipe.vae.enable_tiling()   # 4K解码峰值18.96G→2.6G, 12G必开
    with torch.no_grad():
        lat_cpu = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        del lat
        lat_u = pipe._unpack_latents(lat_cpu, height, width, pipe.vae_scale_factor)
        del lat_cpu
        img = pipe.vae.decode(lat_u, return_dict=False)[0]
        del lat_u
    img = pipe.image_processor.postprocess(img, output_type='pil')[0]
    torch.cuda.empty_cache()
    pipe.vae.to('cpu')
    fn = out_path or f'{OUT}/t2i_{int(time.time())}.png'
    img.save(fn)
    print(f'[t2i3tier] done {time.perf_counter()-t0:.0f}s -> {fn} ({img.size})', flush=True)
    return fn

if __name__ == '__main__':
    img = sys.argv[1] if len(sys.argv) > 1 else r'D:\ai\frontend\assets\chara_qingli.png'
    prompt = sys.argv[2] if len(sys.argv) > 2 else '高清重绘, 保持人物一致'
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 28
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 768
    h = int(sys.argv[6]) if len(sys.argv) > 6 else 1376
    rd = int(sys.argv[7]) if len(sys.argv) > 7 else 12
    w2, h2 = _align_wh(w, h)
    if (w2, h2) != (w, h):
        print(f'[3tier] 尺寸对齐: {w}x{h} -> {w2}x{h2} (64倍数)', flush=True)
    print(f'[3tier] {img} strength={strength} steps={steps} {w2}x{h2} 常驻{rd}双块', flush=True)
    fn = img2img_3tier(prompt, img, strength=strength, steps=steps, width=w, height=h, resident_d=rd)
    print('[3TIER_RESULT]' + fn, flush=True)
