"""
FLUX.2-dev 三层搬运 (从 flux_3tier.py 适配, 结构同思路)
FLUX.2 transformer: 8 双块(1.83G/块) + 48 单块, 总 60G BF16.
差异: setup 用 time_guidance_embed → double/single_stream_mod; RoPE 分 text/img 再 cat;
     双块传 temb_mod_img/temb_mod_txt 返回(enc,hidden); 单块前 cat 文本+图像;
     单块传 temb_mod; 单块后去掉文本token留图像.
用法: python flux2_3tier.py "<prompt>" <steps> <w> <h> <resident_d>
"""
import torch, time, os, sys, struct, json, gc
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


# ── FusedAttention 补丁 (2026-09-09): S>~32768 时 SDPA/flash kernel 对 non-contiguous V 越界 ──
# 根因: 官方 Flux2ParallelSelfAttnProcessor 的 value 是 qkv.chunk 切片视图(non-contiguous),
#      q/k 经 norm+rope 被连续化, v 裸传; 2K 时选中的 kernel 按错误 stride 寻址 → illegal access.
# 修复: 包裹 dispatch_attention_fn, q/k/v 一律 .contiguous() (已连续为 no-op, 无开销).
import diffusers.models.attention_dispatch as _ad
import diffusers.models.transformers.transformer_flux2 as _tf2
_orig_dispatch = _ad.dispatch_attention_fn
def _safe_dispatch(query=None, key=None, value=None, *args, **kwargs):
    if isinstance(query, torch.Tensor): query = query.contiguous()
    if isinstance(key, torch.Tensor): key = key.contiguous()
    if isinstance(value, torch.Tensor): value = value.contiguous()
    return _orig_dispatch(query, key, value, *args, **kwargs)
_ad.dispatch_attention_fn = _safe_dispatch
_tf2.dispatch_attention_fn = _safe_dispatch
MODEL = r'D:\ai\models\FLUX.2-dev'
OUT   = r'D:\ai\output\images'
DEV   = 'cuda'

def _patch_f2(t, resident_d=0):
    nb_d = len(t.transformer_blocks)
    nb_s = len(t.single_transformer_blocks)
    resident_d = max(0, min(resident_d, nb_d))
    for m in (t.x_embedder, t.context_embedder, t.pos_embed, t.norm_out, t.proj_out,
              t.time_guidance_embed, t.double_stream_modulation_img,
              t.double_stream_modulation_txt, t.single_stream_modulation):
        m.to(DEV)
    for i in range(resident_d):
        t.transformer_blocks[i].to(DEV)
    def _slots(blk):
        s = {}
        with torch.no_grad():
            for name, p in blk.named_parameters():
                s[name] = torch.empty_like(p, device=DEV)
        return s
    slots_d = _slots(t.transformer_blocks[resident_d]) if resident_d < nb_d else {}
    slots_s = _slots(t.single_transformer_blocks[0])
    torch.cuda.empty_cache()

    _cpu_backup = {}
    def _backup(blk):
        for name, p in blk.named_parameters():
            _cpu_backup[id(p)] = p.data
    def _restore(blk):
        for name, p in blk.named_parameters():
            p.data = _cpu_backup.pop(id(p), p.data)

    def forward(hidden_states, timestep, guidance=None, encoder_hidden_states=None,
                img_ids=None, txt_ids=None, joint_attention_kwargs=None,
                return_dict=True, kv_cache=None, kv_cache_mode=None,
                num_ref_tokens=0, ref_fixed_timestep=0.0, **kw):
        with torch.no_grad():
            return _fwd(hidden_states, timestep, guidance, encoder_hidden_states,
                        img_ids, txt_ids, joint_attention_kwargs)

    def _fwd(hidden_states, timestep, guidance, encoder_hidden_states,
             img_ids, txt_ids, joint_attention_kwargs):
        num_txt = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = t.time_guidance_embed(timestep, guidance)
        mod_img = t.double_stream_modulation_img(temb)
        mod_txt = t.double_stream_modulation_txt(temb)
        mod_single = t.single_stream_modulation(temb)
        hidden_states = t.x_embedder(hidden_states)
        enc = t.context_embedder(encoder_hidden_states)
        if img_ids is not None and img_ids.ndim == 3: img_ids = img_ids[0]
        if txt_ids is not None and txt_ids.ndim == 3: txt_ids = txt_ids[0]
        irot = t.pos_embed(img_ids)
        trot = t.pos_embed(txt_ids)
        crot = (torch.cat([trot[0], irot[0]], dim=0), torch.cat([trot[1], irot[1]], dim=0))
        jkw = joint_attention_kwargs or {}
        tb = time.perf_counter()   # 块级探针

        # 双块: 逐块, 返回 (enc, hidden)
        for i in range(nb_d):
            block = t.transformer_blocks[i]
            if i < resident_d:
                enc, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=enc,
                                           temb_mod_img=mod_img, temb_mod_txt=mod_txt,
                                           image_rotary_emb=crot, joint_attention_kwargs=jkw)
            else:
                _backup(block)
                for name, p in block.named_parameters():
                    slots_d[name].copy_(p.data, non_blocking=False)
                    p.data = slots_d[name]
                enc, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=enc,
                                           temb_mod_img=mod_img, temb_mod_txt=mod_txt,
                                           image_rotary_emb=crot, joint_attention_kwargs=jkw)
                _restore(block)
        # cat 双块处理后的文本(enc) + 图像(hidden_states) → 单块
        hidden_states = torch.cat([enc, hidden_states], dim=1)

        # 单块: 逐块流式
        for j in range(nb_s):
            block = t.single_transformer_blocks[j]
            _backup(block)
            for name, p in block.named_parameters():
                slots_s[name].copy_(p.data, non_blocking=False)
                p.data = slots_s[name]
            hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=None,
                                  temb_mod=mod_single, image_rotary_emb=crot,
                                  joint_attention_kwargs=jkw)
            _restore(block)
        # 去文本token
        hidden_states = hidden_states[:, num_txt:, ...]

        hidden_states = t.norm_out(hidden_states, temb)
        output = t.proj_out(hidden_states)
        return (output,)

    t.forward = forward
    return resident_d

def _align_wh(w, h, mult=64):
    w = max(mult, round(w / mult) * mult)
    h = max(mult, round(h / mult) * mult)
    return int(w), int(h)

# ═══ text_encoder pread 加载 (照 t5_loader.py, 绕开 safetensors mmap 段错误) ═══
DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32,'U8':torch.uint8}
def _pread_tensor(path, key, hdr_cache):
    """pread 读单个 tensor (struct header + seek, 不 mmap)."""
    if path not in hdr_cache:
        with open(path,'rb') as f:
            hs=struct.unpack('<Q',f.read(8))[0]
            hdr_cache[path]=(8+hs, json.loads(f.read(hs)))
    base, hdr = hdr_cache[path]
    meta = hdr[key]
    o0,o1 = meta['data_offsets']
    with open(path,'rb') as f:
        f.seek(base+o0)
        buf = bytearray(f.read(o1-o0))
    return torch.frombuffer(buf, dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape']).clone()

def load_text_encoder_pread(model_dir, dtype=torch.bfloat16):
    """建空 Mistral3 + pread 逐分片读权重 (照 t5_loader.load_t5_pread), 绕 mmap 段错误.
    返回 Mistral3ForConditionalGeneration (CPU, 权重已注入)."""
    import struct, json, gc
    from transformers import Mistral3ForConditionalGeneration, AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with torch.device('meta'):
        model = Mistral3ForConditionalGeneration(cfg)
    # index.json → key→分片
    idx = json.load(open(f'{model_dir}/model.safetensors.index.json'))
    wmap = idx['weight_map']
    # 按分片分组
    shard_keys = {}
    for k, sh in wmap.items():
        shard_keys.setdefault(sh, []).append(k)
    hdr_cache = {}
    lm_w = {}
    for sh, keys in shard_keys.items():
        print(f'[pread] {sh} ({len(keys)} keys)', flush=True)
        shard_path = f'{model_dir}/{sh}'
        for k in keys:
            if 'language_model.model.' not in k:
                continue   # 跳过 lm_head/vision, 编码只用语言层
            t = _pread_tensor(shard_path, k, hdr_cache)
            rel = k.replace('language_model.model.', '')   # → 'embed_tokens.weight', 'layers.0...'
            lm_w[rel] = t.to(dtype)
        gc.collect()
    print(f'[pread] 读入语言层 {len(lm_w)} keys, 注入 language_model...', flush=True)
    model.model.language_model.load_state_dict(lm_w, strict=False, assign=True)
    del lm_w; gc.collect()
    lm = model.model.language_model
    # 手动初始化 rotary_emb buffer (meta下算出meta, 需CPU重算)
    rope = lm.rotary_emb
    if getattr(rope, 'inv_freq', None) is not None and rope.inv_freq.is_meta:
        cfg = lm.config
        rope_type = cfg.rope_parameters["rope_type"] if hasattr(cfg,'rope_parameters') else "default"
        from transformers.models.mistral.modeling_mistral import ROPE_INIT_FUNCTIONS
        init_fn = ROPE_INIT_FUNCTIONS[rope_type] if rope_type != "default" else rope.compute_default_rope_parameters
        inv_freq, attn_scale = init_fn(cfg, torch.device('cpu'))
        rope.inv_freq = torch.nn.Buffer(inv_freq, persistent=False)
        rope.original_inv_freq = torch.nn.Buffer(inv_freq.clone(), persistent=False)
        print(f'[pread] rotary buffer 已CPU初始化 (rope_type={rope_type})', flush=True)
    # 其余meta (lm_head/vision) 保持不占RAM
    meta_cnt = sum(1 for p in lm.parameters() if p.is_meta)
    print(f'[pread] 语言层注入完成, 剩余meta={meta_cnt}', flush=True)
    return model

def encode_prompt_streaming(pipe, prompt, max_seq=512, out_layers=(10,20,30), device=DEV):
    """逐层流式编码 (照 deep v4flash: 每层上GPU算完释放, 全程 inference_mode).
    text_encoder 45G 塞不下12G → 只驻留当前层, 照 397B/千文 分层."""
    import torch
    from transformers.models.mistral.modeling_mistral import create_causal_mask
    with torch.inference_mode():   # deep 全程 inference_mode (防 autograd/崩)
        # tokenizer 处理 (照官方)
        sys_msg = pipe.system_message if hasattr(pipe, 'system_message') else None
        prompts = [prompt] if isinstance(prompt, str) else prompt
        cleaned = [p.replace("[IMG]", "") for p in prompts]
        messages = [[{"role": "system", "content": sys_msg},
                     {"role": "user", "content": [{"type": "text", "text": p}]}] for p in cleaned]
        inputs = pipe.tokenizer.apply_chat_template(messages, add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors='pt', padding='max_length',
            truncation=True, max_length=max_seq)
        input_ids = inputs['input_ids'].to(device)
        attn = inputs['attention_mask'].to(device)
        lm = pipe.text_encoder.model.language_model
        # embed/norm/rotary 常驻GPU (小)
        lm.embed_tokens.to(device); lm.norm.to(device); lm.rotary_emb.to(device)
        h = lm.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = lm.rotary_emb(h, position_ids=position_ids)
        causal_mask = create_causal_mask(config=lm.config, inputs_embeds=h,
            attention_mask=attn, past_key_values=None, position_ids=position_ids)
        hs = {}
        # 逐层: .to(DEVICE) 算完回CPU (照 deep _build_and_upload 单层驻留)
        for i, layer in enumerate(lm.layers):
            layer.to(device, non_blocking=True)
            new_h = layer(hidden_states=h, attention_mask=causal_mask,
                          position_ids=position_ids, past_key_values=None,
                          use_cache=False, position_embeddings=pos_emb)
            del h
            h = new_h
            del new_h
            if (i + 1) in out_layers:   # 官方 hidden_states[k]=layers[k-1]输出; i从0起 => 存 k=i+1
                hs[i + 1] = h.detach().clone()
            layer.to('cpu')
            torch.cuda.empty_cache()
        del h, pos_emb
        lm.embed_tokens.to('cpu'); lm.norm.to('cpu'); lm.rotary_emb.to('cpu')
        torch.cuda.empty_cache()
    # 取 out_layers 拼接 (照官方逻辑)
    out = torch.stack([hs[k].cpu() for k in out_layers], dim=1)
    out = out.to(pipe.text_encoder.dtype)
    bs, nc, sl, hd = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(bs, sl, nc*hd)
    text_ids = pipe._prepare_text_ids(prompt_embeds)
    return prompt_embeds, text_ids

def text2img_f2(prompt, steps=28, width=768, height=1344, guidance_scale=3.5,
                max_seq=512, out_path=None, resident_d=0, seed=None,
                images=None, embeds_file=None):
    """FLUX.2 三层搬运 文生图 / 图生图(多参考).
    images: PIL.Image 或路径 list (最多~6张参考图, 走官方 _encode_vae_image BN编码后 concat).
    embeds_file: 可选, 直接读现成 embeds 缓存 .pt(含 'embeds' 或 'prompt_embeds'+'text_ids'),
                 跳过 text_encoder 内联编码(避免 transformer+text_encoder 同场内存爆).
    text_encoder逐层流式编码; transformer逐块去噪; GPU tiling解码."""
    width, height = _align_wh(width, height)
    from diffusers import Flux2Pipeline
    from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
    import torch as _t
    import psutil
    t0 = time.perf_counter()
    print('[f2] 加载 pipeline (mmap, RAM不爆)...', flush=True)
    pipe = Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, text_encoder=None)   # 跳过45G text_encoder(用pread流式编码), 懒加载66G transformer
    print(f'[f2] 加载完 {time.perf_counter()-t0:.0f}s', flush=True)
    if _mh:
        _mh.trim('f2.after_load')          # 66GB mmap 已建立: 记一笔并归还闲置页
        pass  # [2026-10-02 主人令] 运行中不再后台清内存
    r = _patch_f2(pipe.transformer, resident_d=resident_d)
    print(f'[f2] patch完 常驻{r}双块', flush=True)

    # ── 阶段1: embeds (流式架构: 编码单独算存缓存, 这里只读, 不内联加载 text_encoder) ──
    import hashlib, os as _os
    prompt_embeds, text_ids = None, None
    # 1) 显式 embeds_file (最优先)
    if embeds_file:
        print(f'[f2] 读 embeds_file {embeds_file}', flush=True)
        ck = _t.load(embeds_file, map_location='cpu', weights_only=True)
        if 'prompt_embeds' in ck:
            prompt_embeds = ck['prompt_embeds'].to(DEV)
            text_ids = ck['text_ids'].to(DEV)
        elif 'embeds' in ck:   # run_f2_full 格式: 只有 embeds, text_ids 现算
            prompt_embeds = ck['embeds'].to(DEV)
            text_ids = pipe._prepare_text_ids(prompt_embeds).to(DEV)
    # 2) 否则查 md5 缓存
    if prompt_embeds is None:
        cache_key = hashlib.md5(f'{prompt}|{max_seq}'.encode()).hexdigest()[:16]
        cache_path = f'D:\\ai\\_cached_flux2_embeds_{cache_key}.pt'
        if _os.path.exists(cache_path):
            print(f'[f2] 命中缓存 {cache_path}', flush=True)
            ck = _t.load(cache_path, map_location='cpu', weights_only=True)
            prompt_embeds = ck['prompt_embeds'].to(DEV)
            text_ids = ck['text_ids'].to(DEV)
    # 3) 都没有 → 报错 (编码必须单独跑, 不能与 transformer 同场全量加载 text_encoder)
    if prompt_embeds is None:
        raise RuntimeError(
            f'[f2] 无 embeds 缓存(prompt={prompt!r})。'
            'text_encoder(45G)不能与 transformer 同场, 请先用独立编码脚本生成 embeds 缓存, '
            '或传 embeds_file= 指向现成 .pt 缓存。'
        )
    print(f'[f2] embeds {tuple(prompt_embeds.shape)} 就绪 {time.perf_counter()-t0:.0f}s', flush=True)
    _t.cuda.empty_cache()

    # ── 阶段2: latent 准备 + 去噪 ──
    nch = pipe.transformer.config.in_channels // 4  # 需确认 FLUX.2 pack
    latents, latent_ids = pipe.prepare_latents(
        1, nch, height, width, prompt_embeds.dtype, DEV,
        _t.Generator(DEV).manual_seed(seed if seed is not None else int(time.time())%10000), None)

    # 图生图: 若给参考图, 编码成 image_latents 并准备 img_ids (官方 prepare_image_latents)
    image_latents, image_latent_ids = None, None
    if images:
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) > 6:
            print(f'[f2] 参考图{len(images)}张>6, 截断到6', flush=True)
            images = images[:6]
        # 转 PIL
        pil_imgs = []
        from PIL import Image as _Image
        for im in images:
            pil_imgs.append(_Image.open(im) if isinstance(im, str) else im)
        # VAE 编码参考图需在 GPU (引擎解码阶段才移VAE, 故这里先移)
        pipe.vae.to(DEV)
        # 参考图 preprocess (官方把参考图放大到生成尺寸→2K时25600token占一半序列, O(S²)烧3倍算力!
        # 改为: 参考图编码分辨率上限1024 (官方多参考机制支持任意尺寸, 坐标独立, 无需对齐生成尺寸)
        imgs_t = []
        ref_h = min(height, 1024)
        ref_w = min(width, 1024)
        for im in pil_imgs:
            if im.size[0] * im.size[1] > 1024 * 1024:
                im = pipe.image_processor._resize_to_target_area(im, 1024 * 1024)
            im = pipe.image_processor.preprocess(im, height=ref_h, width=ref_w, resize_mode="crop")
            imgs_t.append(im)
        # 单图或多图都走官方编码 (VAE BN 反归一化)
        # 必须 inference_mode: VAE 是原生 UNet 未 patch, 不包则 whole graph 扣住全部中间张量 → 24.8G OOM (2026-09-09)
        with _t.inference_mode():
            image_latents, image_latent_ids = pipe.prepare_image_latents(
                imgs_t, batch_size=1, generator=None, device=DEV,
                dtype=pipe.vae.dtype)
        # 编码完 VAE 回 CPU, 腾显存给 transformer
        pipe.vae.to('cpu'); _t.cuda.empty_cache()
        # 释放 VAE 图像暂存
        del imgs_t, pil_imgs
        print(f'[f2] 图生图: {len(images)}张参考图, image_latents{tuple(image_latents.shape)}', flush=True)

    # scheduler timesteps (官方: 用 compute_empirical_mu + 传 sigmas)
    img_seq = (width // 16) * (height // 16)
    mu = compute_empirical_mu(image_seq_len=img_seq, num_steps=steps)
    ts, _ = retrieve_timesteps(pipe.scheduler, steps, DEV,
        sigmas=np.linspace(1.0, 1.0/steps, steps), mu=mu)
    guidance = _t.full([1], guidance_scale, device=DEV, dtype=_t.float32).expand(latents.shape[0])

    for i, t in enumerate(ts):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_model_input = latents.to(pipe.transformer.dtype)
        latent_image_ids = latent_ids
        if image_latents is not None:
            latent_model_input = _t.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
            latent_image_ids = _t.cat([latent_ids, image_latent_ids], dim=1)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input, timestep=timestep/1000, guidance=guidance,
            encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids,
            joint_attention_kwargs=None, return_dict=False)[0]
        noise_pred = noise_pred[:, :latents.size(1)]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        del noise_pred, latent_model_input
        _t.cuda.empty_cache()
        if i % 5 == 0 or i == len(ts)-1:
            print(f'[f2] step {i+1}/{len(ts)} {time.perf_counter()-t0:.0f}s', flush=True)
            if _mh:
                _mh.empty_working_set()    # 每5步归还一次, 防工作集一路涨

    # ── 阶段3: decode (照flux_3tier: GPU解码 + tiling + 逐段del) ──
    pipe.transformer.to('cpu'); _t.cuda.empty_cache()
    pipe.vae.to(DEV)
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()   # 大图解码省显存
    with _t.no_grad():
        lat_u = pipe._unpack_latents_with_ids(latents, latent_ids)
        del latents
        # FLUX.2 VAE = BN 反归一化 + unpatchify (照官方 __call__ line 1016-1026)
        lat_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_u.device, lat_u.dtype)
        lat_bn_std = _t.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(lat_u.device, lat_u.dtype)
        lat_u = lat_u * lat_bn_std + lat_bn_mean
        lat_u = pipe._unpatchify_latents(lat_u)
        img = pipe.vae.decode(lat_u, return_dict=False)[0]
        del lat_u
        img = pipe.image_processor.postprocess(img, output_type='pil')[0]
    _t.cuda.empty_cache()
    pipe.vae.to('cpu')
    fn = out_path or f'{OUT}/flux2_{int(time.time())}.png'
    img.save(fn)
    print(f'[f2] done {time.perf_counter()-t0:.0f}s -> {fn} ({img.size})', flush=True)
    # [2026-10-02] 真正解除 66GB 映射：只有 pipeline 被释放, 虚拟地址空间才还回去
    try:
        pipe.transformer.to('cpu')
        del pipe
        import gc as _gc; _gc.collect(); _gc.collect()
    except Exception:
        pass
    if _mh:
        _mh.trim('f2.done')
    return fn

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'a cute orange cat on white table, photorealistic'
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    rd = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    # 可选第6参: 参考图路径 (逗号分隔, 最多6张), 例: "a.jpg,b.png"
    imgs = None
    if len(sys.argv) > 6 and sys.argv[6].strip():
        imgs = [p.strip() for p in sys.argv[6].split(',') if p.strip()]
    fn = text2img_f2(prompt, steps=steps, width=w, height=h, resident_d=rd, images=imgs)
    print('F2_RESULT:'+fn, flush=True)
