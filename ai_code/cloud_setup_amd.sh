#!/bin/bash
# ==========================================================
# AMD Radeon Cloud 实例初始化 · V4-Flash 自研引擎移植
# 用法:  bash cloud_setup_amd.sh
# 目的:  摸清环境 + 探测 FP8 支持 + 测网速 (不下载、不花时间)
# ==========================================================
set -u

echo "========== 1. GPU / ROCm =========="
rocm-smi --showproductname 2>/dev/null | head -25 || echo "(rocm-smi 不可用)"
echo
rocm-smi --showmeminfo vram 2>/dev/null | head -20 || true

echo
echo "========== 2. Python / PyTorch =========="
python3 - <<'PY' 2>&1
try:
    import torch
    print("torch      :", torch.__version__)
    print("hip        :", getattr(torch.version, "hip", None))
    print("device_cnt :", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_0   :", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch 导入失败:", type(e).__name__, str(e)[:200])
PY

echo
echo "========== 3. ★ FP8 支持探测 (最关键) =========="
python3 - <<'PY' 2>&1
import torch
print("float8_e4m3fn 存在:", hasattr(torch, "float8_e4m3fn"))
try:
    a = torch.ones(4, 4, dtype=torch.float8_e4m3fn, device="cuda")
    print("创建 fp8 张量   : OK")
    b = (a.float() @ a.float().T)
    print("fp8->fp32 计算  : OK", float(b[0, 0]))
except Exception as e:
    print("FP8 失败:", type(e).__name__, str(e)[:200])
# scaled_mm (fp8 矩阵乘常用)
try:
    print("torch._scaled_mm:", "存在" if hasattr(torch, "_scaled_mm") else "不存在")
except Exception as e:
    print("scaled_mm 检查异常:", str(e)[:100])
PY

echo
echo "========== 4. transformers / finegrained_fp8 =========="
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>&1
python3 -c "import transformers.integrations.finegrained_fp8 as m; print('finegrained_fp8: OK ->', m.__file__)" 2>&1

echo
echo "========== 5. 磁盘 / 内存 =========="
df -h 2>/dev/null | grep -E "Filesystem|/workspace|/root|/tmp|overlay" | head -10
echo
free -g 2>/dev/null | head -3
echo "CPU 核心: $(nproc)"

echo
echo "========== 6. 持久盘 /workspace 可写? =========="
if [ -d /workspace ]; then
    touch /workspace/.write_test 2>/dev/null && echo "/workspace 可写 OK" && rm -f /workspace/.write_test \
      || echo "/workspace 不可写 !!"
    df -h /workspace | tail -1
else
    echo "(没有 /workspace 目录 -- 确认持久存储挂载点在哪)"
    ls -la / | head -20
fi

echo
echo "========== 7. 网速 (ModelScope) =========="
timeout 25 curl -o /dev/null -s -w "下载速度: %{speed_download} B/s  (约 %{speed_download} B/s)\n" \
  "https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4-Flash-Base/repo/files" \
  || echo "测速失败/超时"

echo
echo "========== 8. 平台是否预装模型 =========="
ls -la /models/ 2>/dev/null | head -10 || echo "(没有 /models 目录)"

echo
echo "========== 9. pip / 可用包 =========="
python3 -m pip --version 2>&1
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|transformers|aiter|triton|modelscope|accelerate|safetensors) " | head -15

echo
echo "======================================================"
echo "  把以上完整输出发回来，据此决定下载与切分方案"
echo "======================================================"
