#!/bin/bash
# ═══ V4-Flash AMD 实例一键部署 / 恢复 ═══
# 用法(在 Git Bash 里):
#   bash deploy_v4flash_amd.sh <HOST> <PORT> [2|4] [MODEL_REPO]
# 例:
#   bash deploy_v4flash_amd.sh 36.x.x.x 35736 4 deepseek-ai/DeepSeek-V4-Flash
#
# 做四件事:
#   1) 把本地备份里的引擎(已含全部修复)和脚本上传到实例 /workspace
#   2) 装依赖(transformers 5.16.1 等, --no-deps 保护实例自带 ROCm torch)
#   3) 若 /v4flash_fp4 不存在 → 用 hf-mirror 多线程下载(需给 MODEL_REPO)
#   4) 按卡数启动 + 自检(出文本/测速/读盘量)
#
# 前提: 平台的 SSH 公钥已注入实例(或你已 ssh-copy-id)

set -u
HOST="${1:?用法: bash deploy_v4flash_amd.sh <HOST> <PORT> [2|4] [MODEL_REPO]}"
PORT="${2:?缺 PORT}"
CARDS="${3:-2}"
MODEL_REPO="${4:-}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
BK="${BACKUP_DIR:-/d/ai/cloud_v4flash_backup_20260913}"

SSHOPT="-i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
ssh_do() { ssh $SSHOPT root@"$HOST" "$@"; }
scp_do() { scp $SSHOPT "$@"; }

echo "=== 0. 连通性 ==="
ssh_do "hostname; nvidia-smi -L 2>/dev/null || rocm-smi --showmeminfo vram | grep -E 'GPU\[[0-9]\].*(Total|Used)' | head -12"
echo "=== 0b. 卡数确认(应看到 $CARDS 张) ==="

echo "=== 1. 上传引擎 + 脚本 ==="
ssh_do "mkdir -p /workspace/engine"
scp_do "$BK"/engine/*.py root@"$HOST":/workspace/engine/
scp_do "$BK"/scripts/* root@"$HOST":/workspace/
ssh_do "chmod +x /workspace/*.sh; ls /workspace/engine/*.py; ls /workspace/start_*.sh | head"

echo "=== 2. 依赖 (--no-deps 保护 ROCm torch) ==="
ssh_do "/opt/venv/bin/pip install -q --no-deps transformers==5.16.1 huggingface_hub==1.29.0 tokenizers==0.23.1 safetensors==0.8.0 && \
        /opt/venv/bin/python -c 'import torch,transformers;print(\"torch\",torch.__version__,\"hip\",torch.version.hip,\"tf\",transformers.__version__,\"gpus\",torch.cuda.device_count())'"

echo "=== 3. 模型 (/v4flash_fp4) ==="
if ssh_do "test -f /v4flash_fp4/config.json"; then
  echo "  已存在, 跳过下载"; ssh_do "du -sh /v4flash_fp4"
else
  if [ -z "$MODEL_REPO" ]; then
    echo "  ✗ 模型不在且没给 MODEL_REPO —— 请带上仓库名重跑, 例如: ... 4 deepseek-ai/DeepSeek-V4-Flash"
    exit 1
  fi
  echo "  用 hf-mirror 下载 $MODEL_REPO → /v4flash_fp4"
  ssh_do "export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0; \
          /opt/venv/bin/hf download $MODEL_REPO --local-dir /v4flash_fp4 --max-workers 8 2>&1 | tail -5; du -sh /v4flash_fp4"
fi

echo "=== 4. 启动 ($CARDS 卡) ==="
START="/workspace/start_full.sh"; [ "$CARDS" = "4" ] && START="/workspace/start_4card.sh"
ssh_do "pkill -9 -f '[v]4flash_chunked_server'; sleep 2; cd /workspace; setsid nohup bash $START > /workspace/v4f_server.log 2>&1 < /dev/null & sleep 5; echo started"
for i in $(seq 1 30); do
  ssh_do "grep -q HTTP /workspace/v4f_server.log" 2>/dev/null && break
  sleep 5
done
ssh_do "grep -E 'ropefix|dual\]|resident\] 全部|mtp\]|HTTP' /workspace/v4f_server.log | head -6"

echo "=== 5. 自检 ==="
ssh_do "cd /workspace && printf '1+1等于几？' > msg.txt && /opt/venv/bin/python repro2.py 0 16 2>&1 | sed -n 3p | cut -c1-160; \
        echo '--- warn(NaN) 计数:'; grep -c 'warn. probs' /workspace/v4f_server.log; \
        echo '--- 两卡显存:'; rocm-smi --showmeminfo vram 2>/dev/null | grep Used"
echo "=== 完成。测速: ssh -p $PORT root@$HOST 'cd /workspace && /opt/venv/bin/python tpot_test.py' ==="
