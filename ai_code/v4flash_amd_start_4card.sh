#!/bin/bash
# ═══ AMD 云端 V4-Flash 启动脚本（4 卡版，等拿到 4 卡实例直接用）═══
# 布局: GPU0 = 43层常驻(13GB) + hub(3.2GB) + 热池 2500 个(32GB)  ≈ 48GB / 51.5GB
#       GPU1/2/3 = 专家仓库 每卡 3670 个(46.8GB) → 3×3670 = 11010 ≥ 全量 11008 ✓
#       RAM = LRU 只留 1500 暂存（专家已全在显存）
# 计算永远在 GPU0；仓库命中把一个专家(12.75MB)拉回来 ≈0.65ms (P2P ~20GB/s)
# 切换: pkill -9 -f "[v]4flash_chunked_server"; bash start_4card.sh
cd /workspace/engine
export V4F_BASE=/workspace
export V4F_MODEL_DIR=/v4flash_fp4
export V4F_SRC_DIR=/workspace/engine/_v4flash_src
export V4F_TORCH_FP8=1
export V4F_RESIDENT_LAYERS=1
export V4F_GPU_EXP_MAX=2500
export V4F_EXP_STORE_DEV=1,2,3
export V4F_EXP_STORE_MAX=3670
export V4F_LRU_ITEMS=1500
export V4F_KEEP_GPU_CACHE=1
export V4F_MTP=1
export V4F_EXP_WORKERS=6
export V4F_BUILD_WORKERS=6
export V4F_D2H_WORKERS=4
exec /opt/venv/bin/python v4flash_chunked_server.py
