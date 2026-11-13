import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download, hf_hub_download
LOG = r'D:\ai\models\_dl_klein_ltx25.log'
def log(m):
    line = time.strftime('%m-%d %H:%M:%S ') + str(m)
    print(line, flush=True)
    try: open(LOG, 'a', encoding='utf-8').write(line + '\n')
    except Exception: pass

log('=== 开始：FLUX.2 klein 4B (diffusers) ===')
try:
    p = snapshot_download('black-forest-labs/FLUX.2-klein-4B',
        local_dir=r'D:\ai\models\FLUX.2-klein-4B',
        allow_patterns=['transformer/*','text_encoder/*','vae/*','scheduler/*','tokenizer/*','model_index.json','*.json'],
        ignore_patterns=['*.jpg','*.png'],
        max_workers=8)
    log('klein-4B 完成: %s' % p)
except Exception as e:
    log('klein-4B 失败: %s' % str(e)[:250])

log('=== 开始：LTX-2.5 Distilled Q4_K_M (GGUF, 15.7GB) ===')
try:
    p = hf_hub_download('FenomAI/LTX-2.5-Distilled-GGUF', 'LTX-2.5-Distilled-Q4_K_M.gguf',
                        local_dir=r'D:\ai\models\LTX-2.5-Distilled-GGUF')
    log('LTX GGUF 完成: %s' % p)
    for f in ['video_ltx2_5_t2v_GGUF.json','video_ltx2_5_i2v_GGUF.json']:
        try:
            hf_hub_download('FenomAI/LTX-2.5-Distilled-GGUF', f, local_dir=r'D:\ai\models\LTX-2.5-Distilled-GGUF')
            log('工作流 %s OK' % f)
        except Exception as e:
            log('工作流 %s 失败: %s' % (f, str(e)[:90]))
except Exception as e:
    log('LTX GGUF 失败: %s' % str(e)[:250])

log('=== 附：LTX-2.5 官方 VAE（编码器你自有 Gemma 引擎，先不拉 15.4GB 那份）===')
for f in ['vae/ltx-2.5-video-vae-bf16.safetensors', 'vae/ltx-2.5-audio-vae-bf16.safetensors']:
    try:
        hf_hub_download('Lightricks/LTX-2.5', f, local_dir=r'D:\ai\models\LTX-2.5')
        log('  %s OK' % f)
    except Exception as e:
        log('  %s 失败: %s' % (f, str(e)[:120]))
log('=== 全部结束 ===')