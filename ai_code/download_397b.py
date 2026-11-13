"""下载 Qwen3.5-397B-A17B BF16"""
import os, time
from modelscope.hub.file_download import model_file_download

DEST = r"D:\ai\models\qwen3.5-397b-bf16"
MODEL = "Qwen/Qwen3.5-397B-A17B"
os.makedirs(DEST, exist_ok=True)

SMALL = ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","chat_template.jinja","preprocessor_config.json","video_preprocessor_config.json","model.safetensors.index.json","LICENSE","README.md",".gitattributes"]
BIG = [f"model.safetensors-{i:05d}-of-00094.safetensors" for i in range(1,95)]
ALL = SMALL + BIG
done = 0
failed = []

print(f"Downloading {len(ALL)} files to {DEST}")

for i, f in enumerate(ALL):
    dest = os.path.join(DEST, f)
    if os.path.exists(dest):
        sz = os.path.getsize(dest)
        if (f.startswith("model.safetensors") and sz > 8_000_000_000) or (not f.startswith("model.safetensors") and sz > 100):
            done += 1
            continue
    
    tag = "BIG" if f.startswith("model") else "CFG"
    print(f"[{done}/{len(ALL)}] {tag} {f}", flush=True)
    
    ok = False
    for attempt in range(3):
        try:
            t0 = time.time()
            model_file_download(model_id=MODEL, file_path=f, cache_dir=DEST)
            elapsed = time.time() - t0
            sz = os.path.getsize(dest) / 1024**3 if os.path.exists(dest) else 0
            print(f"  OK {sz:.1f}GB {elapsed:.0f}s")
            done += 1
            ok = True
            break
        except Exception as e:
            print(f"  RETRY{attempt+1}: {str(e)[:100]}")
            time.sleep(5)
    
    if not ok:
        failed.append(f)
        print(f"  FAILED")

print(f"\nDONE: {done}/{len(ALL)}")
if failed:
    print(f"FAILED: {failed}")
