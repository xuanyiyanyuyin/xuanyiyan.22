import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'   # 旧 klein 下载 401(xethub CAS) 的解法，必须带
# 本机没设 HTTP(S)_PROXY（直连 hf-mirror），但 NO_PROXY 里的 [::1]/::1 会让 httpx 解析崩
# （Invalid port: ':1]'）。无代理时 NO_PROXY 无意义，清掉避开这个 bug。
for _k in ('NO_PROXY', 'no_PROXY', 'no_proxy'):
    os.environ.pop(_k, None)
from huggingface_hub import snapshot_download

# FLUX.3 Image Klein Base（非蒸馏）bf16 全组件，diffusers 布局。
# 只要 transformer/text_encoder/vae + 配置；排除 variants/(fp8/nvfp4 量化)、model.safetensors(comfy 重复单文件)、图片。
allow = [
    'transformer/*', 'text_encoder/*', 'vae/*', 'tokenizer/*', 'scheduler/*',
    'config.json', 'model_index.json', 'modular_model_index.json', 'manifest.json',
    'ae.safetensors', 'tokenizer_config.json', 'special_tokens_map.json',
]
path = snapshot_download(
    'unsloth/flux-3-image-klein-base',   # 无 gated 镜像（BFL 原仓库 gated:auto 需 token）
    repo_type='model',
    local_dir=r'D:\ai\models\FLUX.3-Klein-Base-bf16',
    allow_patterns=allow,
    max_workers=4,
)
print('下载完成:', path, flush=True)