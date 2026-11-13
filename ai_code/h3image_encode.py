# -*- coding: utf-8 -*-
"""MiniMax-H3-Image 官方 t2i 文本条件编码器（独立子进程，63G 编码器不进主进程内存）

口径逐行照 diffusers main:
  src/diffusers/modular_pipelines/minimax_h3_image/encoders.py
    - MiniMaxH3ImageTextEncoderStep.__call__      (line 145-171)
    - get_qwen3vl_prompt_embeds                  (line 31-100)
  src/diffusers/modular_pipelines/minimax_h3_image/modular_pipeline.py
    - text_encoder_layer = 50
    - max_sequence_length = 4096
    - text_tag = 1

与旧 h3_encode.py（视频 ref2va 任务）的区别，也是质量关键：
  · 视频版呈现是 "<Picture 1>: " + 视觉块 + 提示词，视觉块 tag=0；
  · 官方 t2i 是 **纯文本逐字**：tokenizer(prompt, add_special_tokens=False)，
    无 chat template、无特殊 token、无 vision_inputs，截断 4096，全部行 tag=1。
  · 读 hidden_states[50] 后不走 lm_head；dtype 跟随编码器（bf16）。

用法:
  python h3image_encode.py --prompt "<详细英文画面描述>" --out D:\\ai\\h3image_cond_t2i.pt
  python h3image_encode.py --prompt_file D:\\ai\\prompt.txt --out <pt>
"""
import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

ROOT = r"D:\ai\models\MiniMax-H3-Image"
TEXT_ENCODER = os.path.join(ROOT, "text_encoder")   # 软链 -> MiniMax-H3\Ref2VA\text_encoder
TOKENIZER = os.path.join(ROOT, "tokenizer")
PROCESSOR = os.path.join(ROOT, "processor")
OFFLOAD = r"D:\ai\models\MiniMax-H3\_offload_h3image"

TEXT_ENCODER_LAYER = 50
MAX_SEQUENCE_LENGTH = 4096
TEXT_TAG = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="详细英文画面描述（官方要求整幅画面的长描述）")
    ap.add_argument("--prompt_file", default=None, help="从 UTF-8 文件读提示词")
    ap.add_argument("--out", default=r"D:\ai\h3image_cond_t2i.pt")
    ap.add_argument("--offload", default=OFFLOAD)
    a = ap.parse_args()

    prompt = open(a.prompt_file, encoding="utf-8").read().strip() if a.prompt_file else a.prompt
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("MiniMax-H3-Image 一次只打包一个请求，prompt 必须是非空字符串")
    os.makedirs(a.offload, exist_ok=True)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(PROCESSOR, trust_remote_code=True)
    tokenizer = proc.tokenizer

    # 官方 encoders.py:156-157 —— verbatim, no chat template, no special tokens
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids = token_ids[:MAX_SEQUENCE_LENGTH]
    print("[enc] 官方 t2i 呈现 %d token（逐字 / 无 chat template / 无特殊 token）" % len(token_ids), flush=True)
    print("[enc] prompt: %r" % prompt[:160], flush=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TEXT_ENCODER,
        dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=a.offload,       # accelerate 把放不下的权重落盘
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    dev = model.device
    n_layers = model.config.text_config.num_hidden_layers
    if n_layers <= TEXT_ENCODER_LAYER:
        raise ValueError("编码器只有 %d 层，取不到 hidden_states[50]" % n_layers)
    print("[enc] Qwen3-VL 加载 %.1fs  device=%s  layers=%d" % (
        time.time() - t0, dev, n_layers), flush=True)

    # 官方 get_qwen3vl_prompt_embeds:73-100
    ids = torch.tensor([token_ids], dtype=torch.long, device=dev)
    mm = torch.tensor(proc.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=dev)
    hook = getattr(model, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(model)          # 直接调子模块会绕过 offload 钩子，须手工点一次
    with torch.no_grad():
        out = model.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=mm,
            use_cache=False,
            output_hidden_states=True,
            **{},                        # t2i 无 vision_inputs
        )
    hs = out.hidden_states[TEXT_ENCODER_LAYER].to(dtype=torch.bfloat16).cpu()
    if hook is not None and hasattr(hook, "post_forward"):
        try:
            hook.post_forward(model, {})
        except Exception:
            pass

    tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"hidden_states": hs,
                "input_ids": ids.cpu(),
                "token_tags": tags,
                "prompt": prompt}, a.out)
    print("[enc] 写出 %s  hidden=%s  tag全=%d  %.1fs" % (
        a.out, tuple(hs.shape), TEXT_TAG, time.time() - t0), flush=True)
    print("RESULT: OK", flush=True)


if __name__ == "__main__":
    main()
