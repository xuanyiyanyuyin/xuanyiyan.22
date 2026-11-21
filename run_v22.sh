#!/bin/sh
# 崩溃自愈循环: 非0退出(含SIGSEGV)则3秒后重起, 训练自身从 V22_latest.pt 续训
cd "$(dirname "$0")"
while true; do
  python train_v22_mm.py --data data/pairs_all6_local.tsv --batch_size 320 --epochs 450 "$@"
  code=$?
  echo "[run_v22] exit=$code $(date)"
  [ "$code" -eq 0 ] && break
  sleep 3
done
