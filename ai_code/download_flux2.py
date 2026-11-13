import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
from huggingface_hub import snapshot_download
# 只下 diffusers 需要: transformer分片 + text_encoder分片 + vae/ae + config/scheduler
allow = {
    'transformer/', 'text_encoder/', 'vae/', 'scheduler/', 'tokenizer/',
    'model_index.json', 'ae.safetensors',
}
def keep(p):
    return any(p.startswith(pref) for pref in allow) or p in ('model_index.json','ae.safetensors')
path = snapshot_download(
    'unsloth/FLUX.2-dev',
    repo_type='model',
    local_dir=r'D:\ai\models\FLUX.2-dev',
    allow_patterns=[p for p in [
        'transformer/*', 'text_encoder/*', 'vae/*', 'scheduler/*', 'tokenizer/*',
        'model_index.json', 'ae.safetensors']],
    max_workers=8,   # 并发
)
print('下载完成:', path, flush=True)
