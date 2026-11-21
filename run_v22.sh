#!/usr/bin/env bash
# 崩溃自愈 + OOM自动降档: 非0退出3秒后重起(训练自身从 V22_latest.pt 续训);
# 日志里出现OOM则 batch 降64重试, 最低320
cd "$(dirname "$0")"
BS="${BS:-320}"
while true; do
  python train_v22_mm.py --data data/pairs_all6_local.tsv --batch_size "$BS" --epochs 450 "$@" 2>&1 | tee /tmp/v22_last.log
  code=${PIPESTATUS[0]}
  echo "[run_v22] exit=$code bs=$BS $(date)"
  [ "$code" -eq 0 ] && break
  if grep -q "out of memory\|OutOfMemory" /tmp/v22_last.log; then
    newbs=$((BS - 64)); [ "$newbs" -lt 320 ] && newbs=320
    echo "[run_v22] OOM detected, batch $BS -> $newbs"
    BS=$newbs
  fi
  sleep 3
done
