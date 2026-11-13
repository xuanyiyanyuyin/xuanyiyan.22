#!/bin/bash
# bootstrap.sh —— 拿到新实例后的开机自检 + 一键重建
# 用法: bash bootstrap.sh probe        # 只自检，看丢了什么
#       bash bootstrap.sh rebuild      # 缺什么补什么（下资产/转数据/下模型）
#       bash bootstrap.sh full         # probe + rebuild + 起全参冒烟
set -u
MODE=${1:-probe}

export NO_PROXY=127.0.0.1,localhost,.svc,.svc.cluster.local no_proxy=127.0.0.1,localhost,.svc,.svc.cluster.local
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
export HTTP_PROXY=$P HTTPS_PROXY=$P http_proxy=$P https_proxy=$P
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/opt/aiter
export AITER_TRITON_ONLY=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TMPDIR=/root/bigtmp/work; mkdir -p "$TMPDIR"

RT=/RoboTwin
MR=$RT/experiments/lingbot_vla_v2_6b_robotwin/models
REPRO=/opt/repro

probe() {
  echo "########## 1) 机器与环境 ##########"
  date '+UTC %F %T'; uptime
  nproc; free -g | head -2
  df -h / /workspace 2>&1 | tail -3
  echo "--- GPU ---"; rocm-smi --showmemuse --csv 2>&1 | sed -n '2,3p'
  echo "--- 关键目录 ---"
  for d in /RoboTwin /opt/robotwin-env /opt/lerobot-env /opt/aiter /opt/repro /workspace /models; do
    printf '%-22s %s\n' "$d" "$([ -e $d ] && echo 存在 || echo 缺失)"
  done
  echo "--- 环境变量（非交互 shell 实测）---"
  echo "AITER_TRITON_ONLY=${AITER_TRITON_ONLY:-<空>}  FA_TRITON_AMD=${FLASH_ATTENTION_TRITON_AMD_ENABLE:-<空>}"

  echo "########## 2) 源码 ##########"
  cd $RT 2>/dev/null && {
    git rev-parse HEAD 2>&1
    git -C XPolicyLab rev-parse HEAD 2>&1
    git -C $RT/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD 2>&1
  } || echo "  /RoboTwin 缺失"

  echo "########## 3) 资产 / 数据 ##########"
  echo -n "assets/objects/objaverse/list.json : "; [ -f $RT/assets/objects/objaverse/list.json ] && echo OK || echo 缺
  echo -n "demo_clean 任务数 : "; find $RT/data/demo_clean -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
  echo -n "lerobot   任务数 : "; find $RT/data/lerobot -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
  echo -n "训练清单行数     : "; wc -l < $RT/data/robotwin_demo_clean_joint_v30.txt 2>/dev/null || echo 缺
  echo -n "残留 zip         : "; find $RT/data -name '*.zip' 2>/dev/null | wc -l

  echo "########## 4) 模型 ##########"
  du -sh $MR/* 2>/dev/null
  for f in $MR/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json \
           $MR/robbyant_lingbot-vla-v2-6b/depth/model.pt \
           $MR/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
           $MR/moge-2-vitb-normal/model.pt \
           $MR/Qwen3-VL-4B-Instruct-config-tokenizer/config.json; do
    printf '%-78s %s\n' "$f" "$([ -e $f ] && echo OK || echo 缺)"
  done

  echo "########## 5) 我们的东西（/workspace） ##########"
  du -sh /workspace/records /workspace/tools /workspace/runtime 2>/dev/null
  ls -1 /workspace/records 2>/dev/null | tail -8
  echo -n "merged_lora_9800 : "; du -sh /workspace/runtime/outputs/merged_lora_9800 2>/dev/null || echo 缺
  echo -n "merged_lora_1000 : "; du -sh /root/outputs/merged_lora_1000 2>/dev/null || echo 缺
  echo -n "官方 RoboTwin ckpt : "; du -sh /root/models/robbyant_lingbot-vla-v2-6b-robotwin 2>/dev/null || echo 缺
  echo -n "LoRA checkpoints : "; ls -1 /root/outputs/guard_9800steps_2gpu_193742/checkpoints 2>/dev/null | wc -l
  echo "########## 自检结束 ##########"
}

rebuild() {
  echo "########## 重建开始 $(date '+%T') ##########"
  cd $RT || { echo "!! /RoboTwin 不存在，先 clone"; exit 2; }
  source /opt/robotwin-env/bin/activate 2>/dev/null || true

  if [ ! -f $RT/assets/objects/objaverse/list.json ]; then
    echo ">> [1/4] 下载仿真资产"
    bash scripts/_download_assets.sh
    python3 - <<'PY'
import zipfile, glob, os
for z in glob.glob('/RoboTwin/assets/**/*.zip', recursive=True):
    d = os.path.dirname(z)
    try:
        with zipfile.ZipFile(z) as f: f.extractall(d)
        print('extracted', z)
    except Exception as e: print('ERR', z, e)
PY
  else echo ">> [1/4] 资产已就绪，跳过"; fi

  N=$(find $RT/data/demo_clean -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [ "$N" -lt 50 ]; then
    echo ">> [2/4] 下载 RoboTwin 原始 demo 数据"
    ROBOTWIN_DATA_ROOT=$RT/data \
    HF_ARCHIVE_CACHE=/root/bigtmp/hf-archive \
    HF_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c \
    HF_KEEP_ARCHIVES=0 \
    bash scripts/download_xpolicylab_data.sh
  else echo ">> [2/4] demo_clean 已有 $N，跳过"; fi

  M=$(find $RT/data/lerobot -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  if [ "$M" -lt 50 ]; then
    echo ">> [3/4] LeRobot 转换（8 并行，缓存写大盘 /root/bigtmp）"
    LEROBOT_PYTHON=/opt/lerobot-env/bin/python \
    CONVERSION_JOBS=8 \
    CONVERSION_CACHE_ROOT=/root/bigtmp/robotwin-lerobot-cache \
    bash $REPRO/docker/full/convert_all_data.sh
    find $RT/data/lerobot -mindepth 1 -maxdepth 1 -type d | sed 's#^#/#' > $RT/data/robotwin_demo_clean_joint_v30.txt
    echo "  清单行数: $(wc -l < $RT/data/robotwin_demo_clean_joint_v30.txt)"
  else echo ">> [3/4] lerobot 已有 $M，跳过"; fi

  if [ ! -f $MR/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json ]; then
    echo ">> [4/4] 下载官方基础模型"
    mkdir -p $MR
    /opt/robotwin-env/bin/python -m huggingface_hub.commands.huggingface_cli download \
      robbyant/lingbot-vla-v2-6b --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 \
      --local-dir $MR/robbyant_lingbot-vla-v2-6b
    /opt/robotwin-env/bin/python -m huggingface_hub.commands.huggingface_cli download \
      Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
      --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
      --local-dir $MR/Qwen3-VL-4B-Instruct-config-tokenizer
    /opt/robotwin-env/bin/python -m huggingface_hub.commands.huggingface_cli download \
      Ruicheng/moge-2-vitb-normal --revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368 \
      --local-dir $MR/moge-2-vitb-normal
  else echo ">> [4/4] 基础模型已在，跳过"; fi
  echo "########## 重建结束 $(date '+%T') ##########"
}

case "$MODE" in
  probe) probe ;;
  rebuild) rebuild ;;
  full) probe; rebuild; probe ;;
  *) echo "用法: bootstrap.sh {probe|rebuild|full}"; exit 2 ;;
esac
