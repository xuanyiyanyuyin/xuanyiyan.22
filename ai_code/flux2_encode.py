#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""flux2_encode.py —— FLUX.2 提示词编码（**官方格式版 v2**）

★ v2 修正（2026-09-24，主人指出；两处都能在官方/引擎实现里对照）：
  ① **system message 必须是官方那条**：diffusers `pipelines/flux2/system_messages.py::SYSTEM_MESSAGE`
     （= 引擎里 `pipe.system_message` 的值；源码注明来自 black-forest-labs/flux2 仓库）。
     旧调试脚本 `test_f2_encode.py` 用的是通用 LLM 对话格式 `'You are a helpful assistant'`
     ⇒ 编出的 embeds 不在训练分布上，症状＝**多步也不收敛 / 背景发飘**。
     本脚本**运行时取官方常量**（import → AST 兜底），不手抄，避免换行/空白差异。
  ② **取层下标**：官方 `hidden_states[k] = layers[k-1] 的输出` ⇒ 循环里必须 `if (i+1) in OUT_LAYERS: hs[i+1]=…`
     （照 `flux2_3tier.encode_prompt_streaming` L231）。旧脚本 `if i in (10,20,30)` 是**偏移版**，
     正是档案 L304-308「text_encoder_out_layers 取层偏移修复（根治背景发飘）」里标记的那件事。
  ③ 缓存名带格式标签 `fmt2` ⇒ 与旧（偏移/通用格式）缓存**不冲突**；旧文件保留、不删。

用法:
  python flux2_encode.py "<prompt>" [max_seq=512] [输出路径]
末两行打印 EMBEDS:<路径> / ENCODE_OK
"""
import hashlib
import sys
import time

import torch

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

import flux2_3tier as f2  # noqa: E402

from transformers import AutoProcessor  # noqa: E402
from transformers.models.mistral.modeling_mistral import create_causal_mask  # noqa: E402

OUT_LAYERS = (10, 20, 30)   # 与 flux2_3tier.text2img_f2 的 out_layers 默认值一致
FMT_TAG = 'fmt2'            # 官方格式（SYSTEM_MESSAGE + 取层 i+1）


def official_system_message():
    """官方 system message（优先 import；失败则 AST 解析源码，保证逐字一致）。"""
    try:
        from diffusers.pipelines.flux2.system_messages import SYSTEM_MESSAGE
        return SYSTEM_MESSAGE
    except Exception:
        import ast
        import pathlib
        p = pathlib.Path(r'D:\ai\python\Lib\site-packages\diffusers\pipelines\flux2\system_messages.py')
        tree = ast.parse(p.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'SYSTEM_MESSAGE'
                                                    for t in node.targets):
                return ast.literal_eval(node.value)
        raise RuntimeError('拿不到官方 SYSTEM_MESSAGE')


def cache_path_for(prompt, max_seq=512):
    key = hashlib.md5(f'{prompt}|{max_seq}|{FMT_TAG}'.encode()).hexdigest()[:16]
    return rf'D:\ai\_cached_flux2_embeds_{key}.pt'


def encode(prompt, max_seq=512, out_path=None):
    out_path = out_path or cache_path_for(prompt, max_seq)
    sys_msg = official_system_message()
    print('[f2enc] system_message 长度=%d 前60字=%r' % (len(sys_msg), sys_msg[:60]), flush=True)
    t0 = time.perf_counter()
    tok = AutoProcessor.from_pretrained(f2.MODEL + '/tokenizer', local_files_only=True)
    print('tokenizer OK', flush=True)
    te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
    print(f'pread完成 {time.perf_counter() - t0:.0f}s', flush=True)

    with torch.inference_mode():
        cleaned = prompt.replace('[IMG]', '')
        messages = [[{'role': 'system', 'content': sys_msg},
                     {'role': 'user', 'content': [{'type': 'text', 'text': cleaned}]}]]
        inp = tok.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True, return_dict=True,
            return_tensors='pt', padding='max_length', truncation=True, max_length=max_seq)
        input_ids = inp['input_ids'].to('cuda')
        attn = inp['attention_mask'].to('cuda')

        lm = te.model.language_model
        lm.embed_tokens.to('cuda')
        lm.rotary_emb.to('cuda')
        lm.norm.to('cuda')
        h = lm.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device='cuda').unsqueeze(0)
        pe = lm.rotary_emb(h, position_ids=pos)
        cm = create_causal_mask(config=lm.config, inputs_embeds=h, attention_mask=attn,
                                past_key_values=None, position_ids=pos)
        nl = len(lm.layers)
        if max(OUT_LAYERS) > nl:
            raise RuntimeError(f'层数不足: 需要到层 {OUT_LAYERS}, 实际只有 {nl} 层')
        hs = {}
        for i, layer in enumerate(lm.layers):
            t1 = time.perf_counter()
            layer.to('cuda', non_blocking=True)
            h2 = layer(hidden_states=h, attention_mask=cm, position_ids=pos,
                       past_key_values=None, use_cache=False, position_embeddings=pe)
            del h
            h = h2
            del h2
            # ★ 官方: hidden_states[k] = layers[k-1] 的输出 ⇒ 存 (i+1)
            if (i + 1) in OUT_LAYERS:
                hs[i + 1] = h.detach().clone()
            layer.to('cpu')
            torch.cuda.empty_cache()
            if i % 5 == 0:
                print(f'layer{i} {time.perf_counter() - t1:.1f}s', flush=True)
        out = torch.stack([hs[k].cpu() for k in OUT_LAYERS], dim=1)
        out = out.to(torch.bfloat16)
        bs, nc, sl, hd = out.shape
        pe2 = out.permute(0, 2, 1, 3).reshape(bs, sl, nc * hd)
        torch.save({'embeds': pe2, 'prompt': prompt, 'max_seq': max_seq,
                    'out_layers': list(OUT_LAYERS), 'fmt': FMT_TAG}, out_path)
    print(f'编码完成 {time.perf_counter() - t0:.0f}s embeds={tuple(pe2.shape)}', flush=True)
    print('EMBEDS:' + out_path, flush=True)
    print('ENCODE_OK', flush=True)
    return out_path


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'a cute orange cat'
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    op = sys.argv[3] if len(sys.argv) > 3 else None
    encode(p, ms, op)
