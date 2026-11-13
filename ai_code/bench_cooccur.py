"""
专家共现分析: 找激活聚类 → 打包成组 → 批量SSD读取
用虚拟路由器权重模拟真实激活分布
"""
import torch, torch.nn.functional as F, time, numpy as np
from collections import Counter, defaultdict

DEV='cuda'; H=4096; NUM_EXPERTS=512; TOP_K=10; LAYERS=60
NUM_SAMPLES=200  # 200个虚拟token

print("=" * 60)
print("  MoE 专家共现分析")
print(f"  512专家 × 60层 = 30720实例, Top-{TOP_K}")
print("=" * 60)

# === 虚拟路由器: 模拟真实激活分布 ===
torch.manual_seed(42)
# 每个专家的"亲和度" (模拟训练后的分布)
expert_bias = (torch.randn(NUM_EXPERTS) * 0.5).to(DEV)
layer_shifts = (torch.randn(LAYERS, NUM_EXPERTS) * 0.2).to(DEV)

# === 生成激活数据 ===
print(f"\n[1/3] 生成 {NUM_SAMPLES} 个虚拟token的专家激活...")
t0 = time.perf_counter()

all_activations = []
for _ in range(NUM_SAMPLES):
    hidden = torch.randn(1, H, device=DEV).float()
    for lid in range(LAYERS):
        # 模拟router: hidden → expert logits
        logits = hidden @ torch.randn(H, NUM_EXPERTS, device=DEV).float()
        logits = logits + expert_bias + layer_shifts[lid]
        probs = torch.softmax(logits, -1)
        _, top_indices = torch.topk(probs, TOP_K, -1)
        all_activations.append((lid, top_indices.cpu().numpy().tolist()[0]))

gen_time = time.perf_counter() - t0
print(f"  生成 {len(all_activations)} 个激活记录 ({gen_time:.1f}s)")

# === 共现矩阵 ===
print("\n[2/3] 构建共现矩阵...")
t0 = time.perf_counter()

# A. 层内共现: 哪些专家常一起被选
layer_cooccur = defaultdict(Counter)  # lid → {eid_pair: count}
# B. 跨层共现: 相邻层之间的专家关联
cross_layer_cooccur = Counter()  # {(lid, eid) → (lid+1, eid): count}

for i in range(len(all_activations)):
    lid, experts = all_activations[i]
    # 层内
    for a in range(len(experts)):
        for b in range(a+1, len(experts)):
            pair = tuple(sorted([experts[a], experts[b]]))
            layer_cooccur[lid][pair] += 1
    # 跨层
    if i > 0 and all_activations[i-1][0] == lid - 1:
        prev_experts = set(all_activations[i-1][1])
        for e in experts:
            if e in prev_experts:
                cross_layer_cooccur[(lid-1, lid, e)] += 1

matrix_time = time.perf_counter() - t0
print(f"  构建完成 ({matrix_time:.1f}s)")

# === 分析 ===
print("\n[3/3] 分析结果")
print("-" * 60)

# A. 层内有共现吗？
avg_pairs_per_layer = sum(len(v) for v in layer_cooccur.values()) / LAYERS
max_cooccur = 0
for lid, pairs in layer_cooccur.items():
    if pairs:
        mc = max(pairs.values())
        if mc > max_cooccur: max_cooccur = mc

print(f"\n  [Layer] 层内共现:")
print(f"  每层平均共现对: {avg_pairs_per_layer:.0f}")
print(f"  最大共现次数: {max_cooccur}/{NUM_SAMPLES}")
overlap_rate = max_cooccur / NUM_SAMPLES * 100
print(f"  层内重叠率: {overlap_rate:.1f}%")

# B. 找到最频繁的专家组 (层无关, 只看专家pair)
global_cooccur = Counter()
for lid, pairs in layer_cooccur.items():
    global_cooccur.update(pairs)

top_groups = global_cooccur.most_common(20)
print(f"\n  >> Top 20 高频专家组合 (跨层合并):")
for i, (pair, count) in enumerate(top_groups[:10]):
    print(f"  #{i+1}: 专家{pair[0]:>3d}+{pair[1]:<3d} 共现{count:>4d}次 ({count/NUM_SAMPLES*100:.0f}%)")

# C. 如果将Top组合打包
pack_size = 5  # 每组5个专家
print(f"\n  [PACK] 打包策略 (每组{pack_size}个专家):")
print(f"  当前: 10专家独立读 = 10次随机I/O")
print(f"  打包: N组顺序读 = 更少I/O")

# 计算覆盖率
all_experts_in_top = set()
for (pair, _) in top_groups[:20]:
    all_experts_in_top.add(pair[0])
    all_experts_in_top.add(pair[1])
print(f"  Top 20组合覆盖: {len(all_experts_in_top)}/{NUM_EXPERTS} 专家 ({len(all_experts_in_top)/NUM_EXPERTS*100:.0f}%)")

# D. 跨层连续性
if cross_layer_cooccur:
    top_cross = cross_layer_cooccur.most_common(5)
    print(f"\n  [LINK] 跨层专家连续性:")
    for (l1, l2, eid), count in top_cross:
        print(f"  L{l1}→L{l2} 专家{eid}: {count}次 (可能是attention残留)")

# === 结论 ===
print(f"\n{'='*60}")
print(f"  结论:")
if overlap_rate > 30:
    print(f"  [OK] 共现率{overlap_rate:.0f}% - 专家打包有效!")
    print(f"     建议: 预打包Top专家组合, 常驻RAM缓存")
else:
    print(f"  [WARN] 共现率{overlap_rate:.0f}% - 随机激活为主, 打包收益有限")
    print(f"     建议: 扩大缓存+GB级批量读优先")

# 实际缓存命中率估算
cache_slots = 600
total_instances = LAYERS * NUM_EXPERTS
coverage = cache_slots / total_instances * 100
print(f"\n  当前缓存: {cache_slots}/{total_instances}槽位 = {coverage:.1f}%覆盖率")
print(f"  如果缓存按层分组: 600/{NUM_EXPERTS}={600/NUM_EXPERTS:.1f}层可覆盖")
print(f"  建议: 缓存改为按层分配 (热门层全缓存, 冷门层按需)")
