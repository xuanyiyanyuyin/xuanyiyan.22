#!/bin/bash
# 4 路并行下载 V4-Flash 分片 (hf-mirror)
export HTTP_PROXY='http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128'
export HTTPS_PROXY="$HTTP_PROXY"
export NO_PROXY='127.0.0.1,localhost,.svc,.svc.cluster.local'
R="https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Flash-Base/resolve/main"
cd /v4flash || exit 1

dl() {
  for i in $1; do
    f="model-000${i}-of-00046.safetensors"
    if [ -f "$f" ]; then
      sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
      if [ "$sz" -gt 100000000 ]; then echo "skip $f"; continue; fi
    fi
    t0=$(date +%s)
    curl -sL -C - --retry 6 --retry-delay 3 --max-time 3600 -o "$f" "$R/$f"
    t1=$(date +%s)
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] done $f  ${sz}B  $((t1-t0))s"
  done
}

dl "01 05 09 13 17 21 25 29 33 37 41 45" &
dl "02 06 10 14 18 22 26 30 34 38 42 46" &
dl "03 07 11 15 19 23 27 31 35 39 43" &
dl "04 08 12 16 20 24 28 32 36 40 44" &
wait
echo ALL_DONE
