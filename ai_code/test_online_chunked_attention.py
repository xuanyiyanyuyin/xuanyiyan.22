"""测试在线分块 softmax 注意力与原始 eager 的一致性（面向 1M 上下文）
用 CPU 小规模验证数学正确性
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\ai')

from v4flash_chunked_attention import online_chunked_eager_attention_forward, apply_chunked_attention_patch

import transformers.models.deepseek_v4.modeling_deepseek_v4 as ds4
original_eager = ds4.eager_attention_forward


class MockAttentionModule(nn.Module):
    def __init__(self, num_heads=64, num_key_value_groups=64, head_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.sinks = nn.Parameter(torch.randn(num_heads) * 0.01)
        self.training = False


def test_consistency():
    torch.manual_seed(42)
    print("=== 在线分块 softmax 一致性测试 ===")
    B, H, D = 1, 64, 512

    # 测试不同 Q_len / KV_len 组合（KV 比 Q 长模拟 decode 场景）
    configs = [
        (128, 128),    # prefill 短
        (256, 512),    # prefill 中等，KV>Q
        (512, 1024),   # prefill 长，KV>Q
        (128, 2048),   # decode 长历史
    ]

    for Q_len, KV_len in configs:
        module = MockAttentionModule(num_heads=H, num_key_value_groups=H, head_dim=D)
        scaling = 1.0 / (D ** 0.5)

        query = torch.randn(B, H, Q_len, D)
        key = torch.randn(B, 1, KV_len, D)   # MQA
        value = torch.randn(B, 1, KV_len, D)

        # 因果掩码（Q 只能看前面对应的 KV）
        mask = torch.zeros(B, 1, Q_len, KV_len)
        # 因果：query[i] 只能看 kv[<=i]（简化，KV 可能比 Q 长，末尾对齐）
        offset = KV_len - Q_len
        for i in range(Q_len):
            mask[:, :, i, :i + offset + 1] = 0.0
            mask[:, :, i, i + offset + 1:] = float('-inf')

        # 原始
        out_orig = original_eager(module, query, key, value, mask, scaling, dropout=0.0)

        # 在线分块（不同 chunk 大小）
        for qcs, kcs in [(256, 512), (128, 256)]:
            out_chunk_t = online_chunked_eager_attention_forward(
                module, query, key, value, mask, scaling, dropout=0.0,
                query_chunk_size=qcs, kv_chunk_size=kcs
            )
            max_diff = (out_orig[0] - out_chunk_t[0]).abs().max().item()
            status = "[OK]" if max_diff < 1e-3 else "[FAIL]"
            print(f"  Q={Q_len:4d} KV={KV_len:4d} qcs={qcs:3d} kcs={kcs:3d}: max_diff={max_diff:.3e} {status}")

    print("\n=== 无掩码场景 ===")
    module = MockAttentionModule(num_heads=H, num_key_value_groups=H, head_dim=D)
    scaling = 1.0 / (D ** 0.5)
    Q_len, KV_len = 256, 512
    query = torch.randn(B, H, Q_len, D)
    key = torch.randn(B, 1, KV_len, D)
    value = torch.randn(B, 1, KV_len, D)
    out_orig = original_eager(module, query, key, value, None, scaling, dropout=0.0)
    out_chunk_t = online_chunked_eager_attention_forward(
        module, query, key, value, None, scaling, dropout=0.0,
        query_chunk_size=128, kv_chunk_size=256
    )
    max_diff = (out_orig[0] - out_chunk_t[0]).abs().max().item()
    status = "[OK]" if max_diff < 1e-3 else "[FAIL]"
    print(f"  无mask Q={Q_len} KV={KV_len}: max_diff={max_diff:.3e} {status}")

    print("\n=== 内存峰值对比（KV 全量 vs 分块）===")
    # 用 Q=512, KV=8192 对比 GPU 峰值显存（如果 GPU 可用）
    if torch.cuda.is_available():
        print("  GPU 可用，测试显存...")
        torch.cuda.empty_cache()
        device = 'cuda'
        Q_len, KV_len = 512, 8192
        module = MockAttentionModule(num_heads=H, num_key_value_groups=H, head_dim=D).to(device)
        query = torch.randn(B, H, Q_len, D, device=device)
        key = torch.randn(B, 1, KV_len, D, device=device)
        value = torch.randn(B, 1, KV_len, D, device=device)
        mask = torch.zeros(B, 1, Q_len, KV_len, device=device)

        # 原始（应该 OOM 或占用大）
        try:
            torch.cuda.reset_peak_memory_stats()
            out_orig = original_eager(module, query, key, value, mask, scaling, dropout=0.0)
            peak_orig = torch.cuda.max_memory_allocated() / 1e9
            print(f"  原始: 峰值显存 {peak_orig:.2f} GB")
        except Exception as e:
            print(f"  原始: OOM/失败: {str(e)[:60]}")

        # 在线分块
        torch.cuda.reset_peak_memory_stats()
        out_chunk_t = online_chunked_eager_attention_forward(
            module, query, key, value, mask, scaling, dropout=0.0,
            query_chunk_size=128, kv_chunk_size=512
        )
        peak_chunk = torch.cuda.max_memory_allocated() / 1e9
        print(f"  在线分块: 峰值显存 {peak_chunk:.2f} GB")
    else:
        print("  GPU 不可用，跳过显存测试")

    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    test_consistency()
