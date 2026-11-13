#!/bin/bash
# ═══ AMD 云端 V4-Flash 最终启动脚本（双卡版，当前在用）═══
# 布局: GPU0 = 43层常驻(13GB) + hub(3.2GB) + 热池 2200 个(28GB)
#       GPU1 = 专家仓库 3600 个(46GB)
#       RAM  = LRU 7000 个(89GB)  → 三级合计 ≥ 全量 11008
cd /workspace/engine
export V4F_BASE=/workspace
export V4F_MODEL_DIR=/v4flash_fp4
export V4F_SRC_DIR=/workspace/engine/_v4flash_src
export V4F_TORCH_FP8=1
export V4F_RESIDENT_LAYERS=1
export V4F_GPU_EXP_MAX=2200
export V4F_EXP_STORE_DEV=1
export V4F_EXP_STORE_MAX=3600
export V4F_LRU_ITEMS=7000
export V4F_KEEP_GPU_CACHE=1
export V4F_MTP=1
export V4F_EXP_WORKERS=6
export V4F_BUILD_WORKERS=6
export V4F_D2H_WORKERS=4
exec /opt/venv/bin/python v4flash_chunked_server.py
