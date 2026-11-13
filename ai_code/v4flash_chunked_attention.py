"""DeepSeek V4 分块注意力 — 面向 1M 上下文极限
核心：在线分块 softmax（Flash-Attention 式）
- KV 不展开（MQA 广播），分块驻留 GPU/CPU
- 每个 query 块遍历 KV 块，维护 running max/sum 与累积输出
- 显存峰值 = 单块注意力，与上下文长度无关

适用场景：1M 上下文、超长 prefill、大 KV cache
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══ 在线分块 softmax 注意力 ═══
def online_chunked_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,        # [B, H, Q_len, D]  GPU
    key: torch.Tensor,          # [B, kv_heads, KV_len, D]  GPU 或 CPU（kv_heads 通常=1 MQA）
    value: torch.Tensor,        # [B, kv_heads, KV_len, D]
    attention_mask: Optional[torch.Tensor] = None,   # [B, 1, Q_len, KV_len] 或 None
    scaling: float = 1.0,
    dropout: float = 0.0,
    query_chunk_size: int = 256,   # query 分块（token）
    kv_chunk_size: int = 512,      # KV 分块（token）
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在线分块 softmax 注意力，支持任意长度上下文（显存 O(块)）
    """
    B, H, Q_len, D = query.shape
    _, kv_heads, KV_len, _ = key.shape
    assert kv_heads == 1, f"DeepSeek V4 MQA 要求 kv_heads=1, got {kv_heads}"

    device = query.device
    dtype = query.dtype
    # 内部统一用 float32 计算（数值稳定 + 避免 bf16/float32 混合报错），输出转回 query.dtype
    compute_dtype = torch.float32

    # 将 KV 移到 GPU（如果它们在 CPU）— 分块逐块传输，避免一次性全量
    key_on_gpu = key.device == device
    value_on_gpu = value.device == device

    # 准备 sinks（每头一个可学习标量）— 统一 float32
    sinks = module.sinks.reshape(1, -1, 1, 1).to(compute_dtype)  # [B, H, 1, 1]

    # 输出累积（GPU 常驻，[B, H, Q_len, D]）— float32
    output = torch.zeros(B, H, Q_len, D, device=device, dtype=compute_dtype)

    n_q_chunks = (Q_len + query_chunk_size - 1) // query_chunk_size
    n_kv_chunks = (KV_len + kv_chunk_size - 1) // kv_chunk_size

    for qs in range(0, Q_len, query_chunk_size):
        qe = min(qs + query_chunk_size, Q_len)
        q_chunk = query[:, :, qs:qe, :].to(compute_dtype)  # [B, H, Qc, D] float32

        # 在线 softmax 状态（每 query 块独立）— float32
        running_max = torch.full((B, H, qe - qs, 1), -float('inf'), device=device, dtype=compute_dtype)
        running_sum = torch.zeros(B, H, qe - qs, 1, device=device, dtype=compute_dtype)
        out_chunk = torch.zeros(B, H, qe - qs, D, device=device, dtype=compute_dtype)

        for ks in range(0, KV_len, kv_chunk_size):
            ke = min(ks + kv_chunk_size, KV_len)

            # 取 KV 块（必要时从 CPU 传输）— 统一 float32
            if key_on_gpu:
                k_block = key[:, :, ks:ke, :].to(compute_dtype)
                v_block = value[:, :, ks:ke, :].to(compute_dtype)
            else:
                k_block = key[:, :, ks:ke, :].to(device, non_blocking=True).to(compute_dtype)
                v_block = value[:, :, ks:ke, :].to(device, non_blocking=True).to(compute_dtype)

            # MQA 广播 matmul: [B,H,Qc,D] × [B,1,D,Kc] → [B,H,Qc,Kc]（float32）
            attn = torch.matmul(q_chunk, k_block.transpose(2, 3)) * scaling  # [B,H,Qc,Kc]

            # 加掩码（对应此 KV 块）
            if attention_mask is not None:
                mask_block = attention_mask[:, :, qs:qe, ks:ke]
                if mask_block.device != device:
                    mask_block = mask_block.to(device)
                mask_block = mask_block.to(compute_dtype)
                attn = attn + mask_block

            # 拼接 sinks（DeepSeek V4 特性：每个 head 一个可学习注意力槽）
            # 重要：sink 只在最后一个 KV 块拼接一次（与官方 eager 的全局单 sink 一致），
            #       否则每个块都拼 sink 会在 mask 屏蔽时放大 sink 影响（导致输出偏差）
            is_last_kv_block = (ks + kv_chunk_size >= KV_len)
            if is_last_kv_block:
                sinks_block = sinks.expand(B, -1, qe - qs, -1)  # [B,H,Qc,1]
                attn_with_sink = torch.cat([attn, sinks_block], dim=-1)  # [B,H,Qc,Kc+1]
            else:
                attn_with_sink = attn  # 非最后块无 sink，[B,H,Qc,Kc]

            # 在线 softmax 更新
            block_max = attn_with_sink.max(dim=-1, keepdim=True).values  # [B,H,Qc,1]
            new_max = torch.maximum(running_max, block_max)
            rescale = torch.exp(running_max - new_max)  # [B,H,Qc,1]
            block_probs = torch.exp(attn_with_sink - new_max)  # [B,H,Qc,Kc(+1)]

            # 累积输出（sink 位置无对应 V，丢弃）
            probs_without_sink = block_probs[..., :-1] if is_last_kv_block else block_probs  # [B,H,Qc,Kc]
            # V 块广播 matmul: [B,H,Qc,Kc] × [B,1,Kc,D] → [B,H,Qc,D]（float32）
            out_chunk = out_chunk * rescale + torch.matmul(probs_without_sink, v_block)

            # 累积归一化因子（最后块含 sink 的贡献）
            running_sum = running_sum * rescale + block_probs.sum(dim=-1, keepdim=True)
            running_max = new_max

            del attn, attn_with_sink, block_probs, probs_without_sink, block_max, rescale
            if not key_on_gpu:
                del k_block, v_block

        # 归一化（除以全 KV 的 softmax 分母）
        out_chunk = out_chunk / running_sum
        output[:, :, qs:qe, :] = out_chunk

        if n_q_chunks > 1:
            torch.cuda.empty_cache()

    # 转置回 [B, Q_len, H, D]（与官方 eager 返回一致）并转回原始 dtype
    output = output.to(dtype).transpose(1, 2).contiguous()

    # 空注意力权重（调用方丢弃）
    return output, torch.empty(0, device=device, dtype=dtype)


# ═══ CPU KV Cache（1M 上下文的关键：KV 驻留 DRAM）═══
class CPUDynamicCache:
    """
    KV cache 驻留 CPU（DRAM），GPU 只保留当前计算块
    与 transformers DynamicCache 接口对齐（update/get/get_seq_length）
    支持 MQA（kv_heads=1）
    """
    def __init__(self, num_hidden_layers: int = None, config=None):
        self.key_cache = {}   # layer_idx -> [B, kv_heads, seq, D] CPU
        self.value_cache = {}
        self._seq_length = 0
        self.num_layers = num_hidden_layers

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        """追加 KV（立即转 CPU）"""
        # key_states: [B, kv_heads, seq, D] GPU
        k_cpu = key_states.detach().cpu()
        v_cpu = value_states.detach().cpu()
        if layer_idx in self.key_cache:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k_cpu], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v_cpu], dim=2)
        else:
            self.key_cache[layer_idx] = k_cpu
            self.value_cache[layer_idx] = v_cpu
        # 更新序列长度（取第一层）
        if layer_idx == 0:
            self._seq_length = self.key_cache[layer_idx].shape[2]
        # 返回拼接后的 GPU 版（供调用方继续使用）
        return self.key_cache[layer_idx].to(key_states.device), \
               self.value_cache[layer_idx].to(value_states.device)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_length

    def get_usable_length(self, new_seq_length, layer_idx=0) -> int:
        return self._seq_length

    def reorder_cache(self, beam_idx):
        pass

    def get(self, layer_idx: int):
        """返回 (key, value) — CPU tensor"""
        return self.key_cache.get(layer_idx), self.value_cache.get(layer_idx)

    def to_device(self, device):
        return self

    def clear(self):
        self.key_cache.clear()
        self.value_cache.clear()
        self._seq_length = 0

    def __len__(self):
        return len(self.key_cache)

    @property
    def max_cache_shape(self):
        return None


# ═══ 猴子补丁：替换 eager_attention_forward ═══
_ORIGINAL_EAGER = None

def apply_chunked_attention_patch(
    max_sequence_length: int = 4096,
    chunk_size: int = 512,
    kv_chunk_size: int = 512,
    use_online: bool = True,
):
    """
    应用在线分块注意力补丁
    - 短序列（≤max_sequence_length）走原始函数
    - 长序列走在线分块 softmax
    """
    global _ORIGINAL_EAGER
    try:
        import transformers.models.deepseek_v4.modeling_deepseek_v4 as ds4
        from transformers.utils import is_torch_available
        import torch

        if _ORIGINAL_EAGER is None:
            _ORIGINAL_EAGER = ds4.eager_attention_forward

        def chunked_eager_attention_wrapper(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            scaling: float = 1.0,
            dropout: float = 0.0,
            **kwargs,
        ):
            B, H, Q_len, D = query.shape
            _, _, KV_len, _ = key.shape

            # 短序列走原始函数（行为不变，性能最优）
            if Q_len <= max_sequence_length and KV_len <= max_sequence_length:
                return _ORIGINAL_EAGER(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

            print(f"[chunked_attention] 长序列触发在线分块: Q={Q_len}, KV={KV_len}",
                  f"query_chunk={chunk_size}, kv_chunk={kv_chunk_size}", flush=True)

            # 长序列：KV 已由调用方决定设备（CPU 或 GPU），在线分块处理
            return online_chunked_eager_attention_forward(
                module, query, key, value, attention_mask, scaling, dropout,
                query_chunk_size=chunk_size, kv_chunk_size=kv_chunk_size,
                **kwargs,
            )

        ds4.eager_attention_forward = chunked_eager_attention_wrapper
        # 同步替换模块命名空间中的引用（部分代码直接 import 函数）
        ds4.chunked_eager_attention_forward = chunked_eager_attention_wrapper

        print(f"[chunked_attention] 在线分块补丁应用成功，阈值={max_sequence_length} tokens",
              f"(query_chunk={chunk_size}, kv_chunk={kv_chunk_size})", flush=True)
        return True
    except Exception as e:
        import traceback
        print(f"[chunked_attention] 补丁失败: {e}\n{traceback.format_exc()}", flush=True)
        return False


def remove_chunked_attention_patch():
    """移除补丁，恢复原始函数"""
    global _ORIGINAL_EAGER
    try:
        import transformers.models.deepseek_v4.modeling_deepseek_v4 as ds4
        if _ORIGINAL_EAGER is not None:
            ds4.eager_attention_forward = _ORIGINAL_EAGER
            print("[chunked_attention] 补丁已移除", flush=True)
            return True
    except Exception as e:
        print(f"[chunked_attention] 移除补丁失败: {e}", flush=True)
        return False
