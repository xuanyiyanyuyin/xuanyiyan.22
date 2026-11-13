"""测试分块注意力与原始 eager_attention_forward 的一致性
不加载真实模型，用随机数据模拟模块接口
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\ai')

# 导入分块补丁函数（从 v4flash_chunked_sched 中获取，但避免触发模型加载）
import importlib.util

# 读取 v4flash_chunked_sched.py 中的补丁代码（不执行引擎初始化）
spec = importlib.util.spec_from_file_location("v4flash_chunked_sched", r"D:\ai\v4flash_chunked_sched.py")
# 不加载模块（避免模型初始化），手动提取函数
# 改为直接读取函数定义


# 直接从 transformers 导入原始函数
import transformers.models.deepseek_v4.modeling_deepseek_v4 as ds4
original_eager = ds4.eager_attention_forward
repeat_kv = ds4.repeat_kv


# 从 v4flash_chunked_sched.py 读取分块包装器定义
# 由于 apply_chunked_attention_patch 内部定义 wrapper，我们手动复制核心逻辑
def chunked_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor = None,
    scaling: float = 1.0,
    dropout: float = 0.0,
    chunk_size: int = 512,
    **kwargs
):
    batch_size, num_heads, query_len, head_dim = query.shape
    _, num_kv_heads, kv_len, _ = key.shape

    # 重复 KV
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is None:
        attention_mask = torch.zeros((batch_size, 1, query_len, kv_len),
                                   device=query.device, dtype=query.dtype)

    num_chunks = (query_len + chunk_size - 1) // chunk_size
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(batch_size, -1, 1, 1)

    # 第一阶段：扫描全局最大值
    global_max = torch.tensor(-float('inf'), device='cpu')
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, query_len)
        query_chunk = query[:, :, start:end, :]
        attn_weights = torch.matmul(query_chunk, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, start:end, :]
            attn_weights = attn_weights + mask_chunk
        sinks_chunk = sinks.expand(-1, -1, end - start, -1)
        combined = torch.cat([attn_weights, sinks_chunk], dim=-1)
        chunk_max = combined.max().item()
        global_max = max(global_max, chunk_max)

    global_max_t = torch.tensor(global_max)

    # 第二阶段：计算输出
    all_output_chunks = []
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, query_len)
        chunk_len = end - start

        query_chunk = query[:, :, start:end, :]
        attn_weights = torch.matmul(query_chunk, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, start:end, :]
            attn_weights = attn_weights + mask_chunk

        sinks_chunk = sinks.expand(-1, -1, chunk_len, -1)
        combined = torch.cat([attn_weights, sinks_chunk], dim=-1)
        combined = combined - global_max_t.to(combined.device)
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        attn_probs = probs[..., :-1]
        if dropout > 0 and module.training:
            attn_probs = nn.functional.dropout(attn_probs, p=dropout)

        attn_output_chunk = torch.matmul(attn_probs, value_states)
        attn_output_chunk = attn_output_chunk.transpose(1, 2).contiguous()
        all_output_chunks.append(attn_output_chunk)

    attn_output = torch.cat(all_output_chunks, dim=1)  # [B, S, H, D]（与官方 eager 返回一致）
    return attn_output


# 模拟 DeepseekV4Attention 模块
class MockAttentionModule(nn.Module):
    def __init__(self, num_heads=64, num_key_value_groups=64, head_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.sinks = nn.Parameter(torch.randn(num_heads) * 0.01)
        self.training = False


def run_test():
    torch.manual_seed(42)
    B, H, D = 1, 64, 512
    print("=== 分块注意力一致性测试 ===")

    # 测试不同序列长度
    for S in [128, 512, 1024, 4096]:
        module = MockAttentionModule(num_heads=H, num_key_value_groups=H, head_dim=D)
        scaling = 1.0 / (D ** 0.5)

        # 随机输入（使用 float32 以保证数值精度可对比）
        query = torch.randn(B, H, S, D)
        key = torch.randn(B, 1, S, D)  # MQA: kv_heads=1
        value = torch.randn(B, 1, S, D)

        # 因果掩码
        mask = torch.zeros(B, 1, S, S)
        mask = mask.masked_fill(torch.triu(torch.ones(S, S), diagonal=1).bool().unsqueeze(0).unsqueeze(0), float('-inf'))

        # 原始
        out_orig = original_eager(module, query, key, value, mask, scaling, dropout=0.0)

        # 分块（使用不同 chunk_size）
        for cs in [256, 512]:
            out_chunk = chunked_eager_attention_forward(
                module, query, key, value, mask, scaling, dropout=0.0, chunk_size=cs
            )
            max_diff = (out_orig[0] - out_chunk).abs().max().item()
            rel_diff = ((out_orig[0] - out_chunk).abs() / (out_orig[0].abs() + 1e-8)).max().item()
            status = "[OK]" if max_diff < 1e-3 else "[FAIL]"
            print(f"  S={S:5d} chunk={cs:4d}: max_diff={max_diff:.3e} rel_diff={rel_diff:.3e} {status}")

    print("\n=== 长序列分块（OOM 场景模拟）===")
    # 模拟超长序列：不用真跑 10000（太慢），用 S=8192 但只在 CPU 上跑 chunk 逻辑
    # 这里测试 mask 为 None 的情况
    S = 2048
    module = MockAttentionModule(num_heads=H, num_key_value_groups=H, head_dim=D)
    scaling = 1.0 / (D ** 0.5)
    query = torch.randn(B, H, S, D)
    key = torch.randn(B, 1, S, D)
    value = torch.randn(B, 1, S, D)

    # 无 mask
    out_orig = original_eager(module, query, key, value, None, scaling, dropout=0.0)
    out_chunk = chunked_eager_attention_forward(
        module, query, key, value, None, scaling, dropout=0.0, chunk_size=256
    )
    max_diff = (out_orig[0] - out_chunk).abs().max().item()
    status = "[OK]" if max_diff < 1e-3 else "[FAIL]"
    print(f"  无mask S={S}: max_diff={max_diff:.3e} {status}")

    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    run_test()
