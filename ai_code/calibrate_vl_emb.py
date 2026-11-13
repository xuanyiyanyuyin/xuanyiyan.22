"""
对比官方 embed_tokens 权重 vs 手写 emb, 定位 h0 差异根因
"""
import torch, torch.nn.functional as F, sys, io, json, struct
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

# ═══ 手写 emb ═══
import vl_scorer
scorer=vl_scorer.VLScorer()
mine_emb=scorer.emb  # [V,2048]
print(f'[calib] mine emb {tuple(mine_emb.shape)}')

# ═══ 官方 embed_tokens ═══
from transformers import Qwen3VLMoeForConditionalGeneration
official=Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0:'6GiB','cpu':'22GiB'}, trust_remote_code=True,
    local_files_only=True, offload_folder=r'D:\ai\offload')
ref_emb=official.model.language_model.embed_tokens.weight
print(f'[calib] official emb {tuple(ref_emb.shape)}')
# 对比
a=mine_emb.detach().float().cpu(); b=ref_emb.float().cpu()
d=(a-b).abs()
print(f'[calib] emb diff max_abs={d.max().item():.4f} rel={d.norm()/b.norm().item():.6f}')
# 看是否完全一致
print(f'[calib] identical: {torch.allclose(a,b,atol=1e-3)}')
# lm_head vs emb (tie?)
print(f'[calib] official lm_head exists: {hasattr(official,"lm_head")}')
if hasattr(official,'lm_head'):
    lh=official.lm_head.weight
    print(f'[calib] lm_head shape {tuple(lh.shape)}')
    print(f'[calib] lm_head==emb: {torch.allclose(lh.float().cpu(), b, atol=1e-3)}')
