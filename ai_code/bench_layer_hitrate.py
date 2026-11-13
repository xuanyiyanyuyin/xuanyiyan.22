"""
分层缓存命中率统计: 验证前10层/后10层是否比中间层有更高重叠率
"""
import torch, torch.nn.functional as F, time, numpy as np
from collections import defaultdict, Counter

DEV='cuda'; H=4096; NUM_EXPERTS=512; TOP_K=10; LAYERS=60
NUM_SAMPLES=500  # 500 tokens

torch.manual_seed(42)
expert_bias = (torch.randn(NUM_EXPERTS) * 0.5).to(DEV)
layer_shifts = (torch.randn(LAYERS, NUM_EXPERTS) * 0.2).to(DEV)

print("=" * 60)
print("  [Test 1] 分层缓存命中率分析")
print(f"  {LAYERS}层 x {NUM_SAMPLES}tokens, Top-{TOP_K}")
print("=" * 60)

# === 生成激活 ===
print("\n[1/2] Generate activations...")
t0 = time.perf_counter()

# layer_activations[lid] = set of expert IDs activated across all tokens
layer_activations = defaultdict(set)
# layer_history[lid][token_i] = list of expert IDs
layer_history = defaultdict(list)

for ti in range(NUM_SAMPLES):
    hidden = torch.randn(1, H, device=DEV).float()
    for lid in range(LAYERS):
        logits = hidden @ torch.randn(H, NUM_EXPERTS, device=DEV).float()
        logits = logits + expert_bias + layer_shifts[lid]
        probs = torch.softmax(logits, -1)
        _, top_indices = torch.topk(probs, TOP_K, -1)
        eids = top_indices.cpu().numpy().tolist()[0]
        for e in eids: layer_activations[lid].add(e)
        layer_history[lid].append(set(eids))

print(f"  Done ({time.perf_counter()-t0:.1f}s)")

# === 分层重叠率 ===
print("\n[2/2] Per-layer overlap rate:")
print("-" * 60)
print(f"{'区间':10s} {'层':>4s} | {'唯一专家数':>8s} | {'重叠率':>8s} | {'评级'}")
print("-" * 60)

zones = [("前10层", range(0,10)), ("中40层", range(10,50)), ("后10层", range(50,60))]

for zone_name, layer_range in zones:
    zone_experts = set()
    zone_overlaps = 0
    zone_tokens = 0

    for lid in layer_range:
        zone_experts |= layer_activations[lid]
        # 计算层内重叠: token间专家交集
        for ti in range(NUM_SAMPLES - 1):
            overlap = len(layer_history[lid][ti] & layer_history[lid][ti+1])
            zone_overlaps += overlap
            zone_tokens += TOP_K

    unique = len(zone_experts)
    overlap_pct = zone_overlaps / zone_tokens * 100 if zone_tokens > 0 else 0

    rating = "HOT" if overlap_pct > 20 else ("WARM" if overlap_pct > 10 else "COLD")
    print(f"{zone_name:10s} {min(layer_range):>2d}-{max(layer_range):<2d} | {unique:>8d} | {overlap_pct:7.1f}% | {rating}")

# === 每层详细 ===
print(f"\n  逐层重叠率 (每层):")
for lid in range(LAYERS):
    overlaps = 0
    for ti in range(NUM_SAMPLES - 1):
        overlap = len(layer_history[lid][ti] & layer_history[lid][ti+1])
        overlaps += overlap
    pct = overlaps / (NUM_SAMPLES - 1) / TOP_K * 100
    bar = "#" * int(pct // 2)
    mark = " <<<" if lid < 10 or lid >= 50 else ""
    print(f"  L{lid:02d}: {pct:5.1f}% {bar}{mark}")

# === 缓存分配建议 ===
print(f"\n{'='*60}")
print(f"  缓存分配建议 (600槽位):")
total_slots = 600
for zone_name, layer_range in zones:
    zone_experts = set()
    for lid in layer_range: zone_experts |= layer_activations[lid]
    unique = len(zone_experts)

    if zone_name == "前10层":
        slots = int(total_slots * 0.35)
    elif zone_name == "后10层":
        slots = int(total_slots * 0.25)
    else:
        slots = int(total_slots * 0.40)

    coverage = slots / unique * 100 if unique > 0 else 0
    print(f"  {zone_name}: {slots}槽/{unique}唯一专家 = {coverage:.1f}%覆盖")

print(f"\n  结论: 如果前后层重叠率>中间层, 加权分配有效.")
print(f"        如果所有层重叠率接近, LRU已是最优.")
