#!/bin/bash
# 用 curl 直接拉 V4-Flash 分片 (hf-mirror, 支持断点续传)
export HTTP_PROXY='http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128'
export HTTPS_PROXY="$HTTP_PROXY"
export NO_PROXY='127.0.0.1,localhost,.svc,.svc.cluster.local'
R="https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash-Base/resolve/main"
cd /v4flash || exit 1

for i in $(seq -w 1 46); do
  f="model-000${i}-of-00046.safetensors"
  if [ -f "$f" ]; then
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 100000000 ]; then echo "skip $f ($sz)"; continue; fi
  fi
  t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] start $f"
  curl -sL -C - --retry 8 --retry-delay 5 --max-time 1800 -o "$f" "$R/$f"
  t1=$(date +%s)
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M:%S)] done $f  $(echo "scale=1; $sz/1073741824" | bc)GB  $((t1-t0))s  $(echo "scale=1; $sz/($t1-$t0+1)/1048576" | bc)MB/s"
done
# 非权重文件
for f in config.json tokenizer.json tokenizer_config.json generation_config.json .gitattributes; do
  [ -f "$f" ] && continue
  curl -sL -C - --retry 5 -o "$f" "$R/$f" 2>/dev/null
done
echo ALL_DONE
