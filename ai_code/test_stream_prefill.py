"""流式 prefill 端到端测试：迷你 DeepseekV4 模型对比「全量 prefill」vs「流式 prefill」
用官方 DeepseekV4DecoderLayer + 随机权重，验证数学等价性
"""
import sys, json
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\ai')

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
import transformers.models.deepseek_v4.modeling_deepseek_v4 as ds4
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask


def make_mini_config():
    return DeepseekV4Config(
        vocab_size=320, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, sliding_window=8, hc_mult=2, o_groups=2, o_lora_rank=4,
        compress_rates={'compressed_sparse_attention': 4, 'heavily_compressed_attention': 8},
        moe_intermediate_size=32, num_experts=4, num_experts_per_tok=1,
        shared_expert_num_experts=1, layer_types=['sliding_attention', 'sliding_attention'],
        max_position_embeddings=1000, rms_norm_eps=1e-5, rope_theta=10000.0,
    )


class MiniV4Engine:
    """模拟引擎 forward：embedding + layers + norm + lm_head（照搬 v4flash_sched.py forward 逻辑）"""
    def __init__(self, cfg, seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.embed_w = torch.randn(cfg.vocab_size, cfg.hidden_size)
        self.norm_w = torch.randn(cfg.hidden_size)
        self.lm_head_w = torch.randn(cfg.vocab_size, cfg.hidden_size)

        # 官方层（sliding_attention，compressor=None）
        self.layers = nn.ModuleList([
            ds4.DeepseekV4DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        ])
        self.rotary_emb = ds4.DeepseekV4RotaryEmbedding(cfg)
        # 用空超连接头（hc_mult 简化：identity 展开）
        self.hc_mult = cfg.hc_mult

    def forward(self, input_ids, past_key_values=None):
        cfg = self.cfg
        B, S = input_ids.shape
        device = input_ids.device

        if past_key_values is None:
            cache = DynamicCache(config=cfg)
            position_ids = torch.arange(S, device=device).unsqueeze(0)
            pkv_for_mask = None
        else:
            cache = past_key_values
            past_len = cache.get_seq_length()
            if S > 1:
                position_ids = torch.arange(past_len, past_len + S, device=device).unsqueeze(0)
            else:
                position_ids = torch.tensor([[past_len]], device=device)
            pkv_for_mask = cache

        h = F.embedding(input_ids, self.embed_w)
        h = h.unsqueeze(2).expand(B, S, self.hc_mult, -1).contiguous()  # [B,S,hc_mult,D]
        h_ref = h[:, :, 0, :]
        pe = {
            'main': self.rotary_emb(h_ref, position_ids=position_ids, layer_type='main'),
            'compress': self.rotary_emb(h_ref, position_ids=position_ids, layer_type='compress'),
        }
        causal_mask = create_sliding_window_causal_mask(
            cfg, inputs_embeds=h_ref, attention_mask=None,
            past_key_values=pkv_for_mask, position_ids=position_ids)

        for lid, layer in enumerate(self.layers):
            h = layer(h, position_embeddings=pe, position_ids=position_ids,
                      attention_mask=causal_mask, input_ids=input_ids, past_key_values=cache)

        # 超连接头简化：取第一个通道（hc_head 在完整引擎中处理）
        h_main = h[:, :, 0, :]  # [B,S,D]
        h_main = F.rms_norm(h_main.float(), (cfg.hidden_size,), self.norm_w.float(), cfg.rms_norm_eps)
        logits = h_main @ self.lm_head_w.T
        return logits, cache


def run_test():
    print("=== 流式 prefill 一致性测试（迷你 DeepseekV4）===")
    cfg = make_mini_config()
    cfg._attn_implementation = 'eager'
    engine = MiniV4Engine(cfg, seed=42)

    torch.manual_seed(123)
    total_len = 20
    input_ids = torch.randint(10, 300, (1, total_len))  # [1, 20]

    # === 全量 prefill（一次）===
    logits_full, cache_full = engine.forward(input_ids)
    print(f"全量 prefill: logits shape={tuple(logits_full.shape)}")

    # === 流式 prefill（分段 8+8+4）===
    segments = [8, 8, 4]
    cache = None
    logits_last = None
    pos = 0
    for seg_len in segments:
        seg = input_ids[:, pos:pos+seg_len]
        logits_seg, cache = engine.forward(seg, past_key_values=cache)
        logits_last = logits_seg[:, -1, :]  # 每段最后一个 token 的 logits
        print(f"  段 [{pos}:{pos+seg_len}] logits={tuple(logits_seg.shape)} cache_len={cache.get_seq_length()}")
        pos += seg_len

    # 全量的最后一个 token logits
    logits_full_last = logits_full[:, -1, :]

    max_diff = (logits_full_last - logits_last).abs().max().item()
    rel_diff = ((logits_full_last - logits_last).abs() / (logits_full_last.abs() + 1e-8)).max().item()
    status = "[OK]" if max_diff < 1e-4 else "[FAIL]"
    print(f"\n最后 token logits 对比: max_diff={max_diff:.3e} rel_diff={rel_diff:.3e} {status}")

    # 也对比所有中间 token（前 8+8 段是否与全量一致）
    print("\n各段内部 token 也一致吗?")
    cache = None
    pos = 0
    for seg_len in segments:
        seg = input_ids[:, pos:pos+seg_len]
        logits_seg, cache = engine.forward(seg, past_key_values=cache)
        full_seg = logits_full[:, pos:pos+seg_len, :]
        d = (full_seg - logits_seg).abs().max().item()
        s = "[OK]" if d < 1e-4 else "[FAIL]"
        print(f"  段 [{pos}:{pos+seg_len}]: max_diff={d:.3e} {s}")
        pos += seg_len

    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    import torch.nn as nn
    run_test()
