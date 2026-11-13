import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'          # ★ 关掉 Xet，避免 401
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download
LOG = r'D:\ai\models\_dl_klein_ltx25.log'
def log(m):
    line = time.strftime('%m-%d %H:%M:%S ') + str(m)
    print(line, flush=True)
    try: open(LOG, 'a', encoding='utf-8').write(line + '\n')
    except Exception: pass
log('=== klein-4B 重试（HF_HUB_DISABLE_XET=1）===')
try:
    p = snapshot_download('black-forest-labs/FLUX.2-klein-4B',
        local_dir=r'D:\ai\models\FLUX.2-klein-4B',
        allow_patterns=['transformer/*','text_encoder/*','vae/*','scheduler/*','tokenizer/*','model_index.json','*.json'],
        ignore_patterns=['*.jpg','*.png'],
        max_workers=8)
    log('klein-4B 完成: %s' % p)
except Exception as e:
    log('klein-4B 仍失败: %s' % str(e)[:300])