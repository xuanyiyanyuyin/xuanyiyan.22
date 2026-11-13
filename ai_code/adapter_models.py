"""
模型适配器: 将397B技术(mmap+V22读写头)应用到新模型
适配: Qwen3.6-35B / Qwen3-VL-30B / Qwen3.5-122B
"""
import os, sys, json, time
sys.path.insert(0, r"D:\ai")
from expert_reader import SafeExpertReader
from pipeline_core import MODELS, create_memory_store

print("=" * 60)
print("  模型适配: 397B技术 → AI动画管线")
print("=" * 60)


def check_model(model_id: str):
    """检查模型下载状态, 验证mmap可用"""
    cfg = MODELS[model_id]
    path = cfg["dir"]
    if not os.path.exists(path):
        return "NOT_DOWNLOADED"

    # 检查safetensors文件
    sfs = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    idx = os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    size = sum(os.path.getsize(os.path.join(path, f))
              for f in sfs) / 1024**3

    if idx and sfs:
        return f"READY ({size:.0f}GB, {len(sfs)} shards)"
    elif sfs:
        return f"PARTIAL ({size:.0f}GB, no index)"
    return "INCOMPLETE"


def init_expert_reader(model_id: str):
    """初始化mmap专家读取器"""
    cfg = MODELS[model_id]
    try:
        reader = SafeExpertReader(cfg["dir"])
        return reader
    except Exception as e:
        print(f"  [FAIL] {model_id}: {e}")
        return None


def init_memory_store(model_id: str):
    """初始化V22读写头"""
    try:
        memories = create_memory_store(model_id)
        return memories
    except Exception as e:
        print(f"  [WARN] Memory init: {e}")
        return {}


# ═══════════ 检查所有模型 ═══════════
print("\n[1/3] 模型下载状态:")
for mid in MODELS:
    status = check_model(mid)
    mark = "OK" if "READY" in status else ("..." if "PARTIAL" in status else "XX")
    print(f"  [{mark}] {mid:20s}: {status}")

# ═══════════ 初始化推理引擎 ═══════════
print("\n[2/3] 初始化mmap专家读取器:")
readers = {}
for mid in MODELS:
    if "READY" in check_model(mid):
        r = init_expert_reader(mid)
        if r:
            readers[mid] = r
            print(f"  [OK] {mid} → mmap ready")

# ═══════════ 初始化V22读写头 ═══════════
print("\n[3/3] 初始化V22读写头:")
for mid in readers:
    mem = init_memory_store(mid)
    if mem:
        print(f"  [OK] {mid} → {len(mem)} layers, V22 memory active")

# ═══════════ 总结 ═══════════
print(f"\n{'='*60}")
print(f"  就绪模型: {len(readers)}/{len(MODELS)}")
print(f"  技术栈:")
print(f"    - mmap虚拟内存 (expert_reader.py v2)")
print(f"    - V22读写头 (memory_head.py)")
print(f"    - Scattered MoE管道 (pipeline_core.py)")
print(f"    - 零量化 BF16/FP16")
print(f"\n  待启动: 模型权重 → 管线适配 → ComfyUI桥接")
