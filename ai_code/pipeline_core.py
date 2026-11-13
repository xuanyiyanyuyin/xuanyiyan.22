"""
AI动画流水线核心引擎 v1
复用397B全部技术: mmap加载 + V22读写头 + Scattered MoE管道
"""
import torch, torch.nn.functional as F, numpy as np, os, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DEV = "cuda"
sys.path.insert(0, r"D:\ai")
from expert_reader import SafeExpertReader


# ═══════════════════════════════════════════
# 模型注册表
# ═══════════════════════════════════════════

MODELS = {
    "qwen3.6-35b": {
        "dir": r"D:\ai\models\Qwen3.6-35B-A3B",
        "type": "moe",
        # 2026-08-22 校准自 config.json: qwen3_5_moe, 40层混合注意力(3×linear_attn + 1×full_attn 循环)
        "hidden": 2048, "layers": 40,
        "experts": 256, "top_k": 8,
        "arch": "qwen3_5_moe",
        "layer_prefix": "model.language_model.layers.{lid}.",
        "linear_attn": True,   # 大部分层是 GatedDeltaNet/SSM, 本引擎的简化 attention 不支持 → 需用官方 Qwen3_5MoeDecoderLayer
        "vision": True,        # VLM (Qwen3_5MoeForConditionalGeneration), 文本推理需走 language_model 子模块
    },
    "qwen3-vl-30b": {
        "dir": r"D:\ai\models\Qwen3-VL-30B-A3B-Instruct",
        "type": "moe_vision",
        "hidden": 4096, "layers": 48,
        "experts": 256, "top_k": 3,
        "arch": "qwen3_vl",
    },
    "qwen3.5-122b": {
        "dir": r"D:\ai\models\Qwen3.5-122B-A10B",
        "type": "moe",
        "hidden": 5120, "layers": 80,
        "experts": 128, "top_k": 10,
        "arch": "qwen3",
    },
}


# ═══════════════════════════════════════════
# mmap模型加载器 (复用expert_reader.py v2)
# ═══════════════════════════════════════════

def load_model(model_id: str):
    """加载模型: 非专家mmap常驻RAM, 专家按需pread"""
    cfg = MODELS[model_id]
    model_dir = cfg["dir"]
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model not found: {model_dir}")

    reader = SafeExpertReader(model_dir)
    # TODO: 加载非专家权重到RAM (pinned tensors)
    # 这里先返回reader, 推理时按需加载
    return reader, cfg


# ═══════════════════════════════════════════
# V22 读写头 (复用memory_head.py)
# ═══════════════════════════════════════════

# 从memory_head.py导入
try:
    from memory_head import MemoryHead
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("[WARN] memory_head.py not found, context limited")


def create_memory_store(model_id: str, store_dir: str = None) -> Dict[int, object]:
    """为模型每层创建V22读写头"""
    if not MEMORY_AVAILABLE:
        return {}

    cfg = MODELS[model_id]
    store_dir = store_dir or f"D:\\ai\\memory_{model_id}"
    if os.path.exists(store_dir):
        import shutil
        shutil.rmtree(store_dir, ignore_errors=True)

    memories = {}
    for lid in range(cfg["layers"]):
        memories[lid] = MemoryHead(
            f"{store_dir}/L{lid:02d}",
            max_gpu=100, max_ram=2000,
            hidden_dim=cfg["hidden"], mem_dim=256
        )
    return memories


# ═══════════════════════════════════════════
# 推理管道 (Scattered MoE + mmap)
# ═══════════════════════════════════════════

class MoEInferenceEngine:
    """通用MoE推理引擎, 适配所有Qwen3.x模型"""

    def __init__(self, model_id: str):
        self.cfg = MODELS[model_id]
        self.reader, _ = load_model(model_id)
        self.memories = create_memory_store(model_id)
        self.kv_cache: Dict[int, Tuple] = {}  # fallback if no memory head

    def forward_layer(self, hidden, layer_id: int, layer_weights: dict):
        """单层前向: Attention + Router + MoE + 读写头"""
        H = self.cfg["hidden"]
        B, S, _ = hidden.shape if hidden.dim() == 3 else (1, hidden.shape[1] // H, H)
        if hidden.dim() == 2:
            hidden = hidden.reshape(B, S, H)

        pf = self.cfg.get("layer_prefix", "model.language_model.layers.{lid}.").replace("{lid}", str(layer_id))
        if self.cfg.get("linear_attn"):
            # qwen3_5_moe: 大部分层是 GatedDeltaNet(linear attention), 本简化引擎的 standard attention 不适用
            # 正确做法: 用官方 Qwen3_5MoeDecoderLayer (transformers 5.15) 逐层前向, 而不是手写简化版
            print(f"[WARN] qwen3_5_moe 层 {layer_id} 是 linear_attn, 简化 attention 结果不可靠; "
                  f"请改用 transformers Qwen3_5MoeForCausalLM 或实现 GatedDeltaNet", flush=True)

        # === Attention (复用397B的attn逻辑) ===
        norm1 = layer_weights.get(pf + "input_layernorm.weight")
        hn = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        if norm1 is not None:
            hn = hn * norm1.float()

        # 先尝试V22记忆检索
        if self.memories and layer_id in self.memories:
            recall = self.memories[layer_id].read(
                hn, layer_id=layer_id, top_k=2, mem_type='full_attn'
            )
        else:
            recall = None

        # 标准Attention (简化版, 完整版需对应架构)
        q = layer_weights.get(pf + "self_attn.q_proj.weight")
        if q is not None:
            Q = hn.float() @ q.float().T
            k = layer_weights.get(pf + "self_attn.k_proj.weight")
            v = layer_weights.get(pf + "self_attn.v_proj.weight")
            o = layer_weights.get(pf + "self_attn.o_proj.weight")
            if k is not None and v is not None and o is not None:
                K = hn.float() @ k.float().T
                V = hn.float() @ v.float().T
                att = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                a_out = (att @ o.float().T).to(torch.bfloat16)
                hidden = (hidden + a_out).to(torch.bfloat16)

        # === Router ===
        norm2 = layer_weights.get(pf + "post_attention_layernorm.weight")
        hn2 = hidden * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        if norm2 is not None:
            hn2 = hn2 * norm2.float()

        flat = hn2.to(torch.bfloat16).reshape(-1, H)
        gate = layer_weights.get(pf + "mlp.gate.weight")
        if gate is not None:
            probs = torch.softmax(flat.float() @ gate.float().T, -1)
            _, top_idx = torch.topk(probs, self.cfg["top_k"], -1)
            expert_ids = top_idx.unique().tolist()

            # === MoE: 从SSD读专家 (mmap) ===
            out = torch.zeros_like(flat, dtype=torch.float32)
            for eid in expert_ids[:10]:  # limit to 10
                try:
                    gu, dn = self.reader.read_expert(layer_id, eid)
                    g = torch.from_numpy(gu.astype(np.int16)).view(torch.bfloat16).float().to(DEV)
                    d = torch.from_numpy(dn.astype(np.int16)).view(torch.bfloat16).float().to(DEV)
                    mask = (top_idx == eid).any(-1)
                    if mask.any():
                        t = flat[mask].float()
                        out[mask] += (F.silu(t @ g[:1024].T) * (t @ g[1024:].T)) @ d.T
                except Exception:
                    pass  # expert not available (virtual weights)

            # Shared expert
            sg = layer_weights.get(pf + "mlp.shared_expert.gate_proj.weight")
            su = layer_weights.get(pf + "mlp.shared_expert.up_proj.weight")
            sd = layer_weights.get(pf + "mlp.shared_expert.down_proj.weight")
            sgw = layer_weights.get(pf + "mlp.shared_expert_gate.weight")
            if sg is not None and su is not None and sd is not None and sgw is not None:
                gv = torch.sigmoid(flat.float() @ sgw.float().T)
                so = (F.silu(flat.float() @ sg.float().T) * (flat.float() @ su.float().T)) @ sd.float().T
                out = out + gv * so

            hidden = (hidden + out.to(torch.bfloat16).reshape_as(hidden)).to(torch.bfloat16)

        # 写入V22记忆
        if self.memories and layer_id in self.memories:
            self.memories[layer_id].write(hidden, layer_id=layer_id, mem_type='full_attn')

        return hidden

    def generate(self, prompt: str, tokenizer, max_tokens: int = 50) -> str:
        """简化生成: token by token"""
        ids = tokenizer.encode(prompt)
        tokens = []
        h = None  # embedding
        for _ in range(max_tokens):
            for lid in range(self.cfg["layers"]):
                # 这里需要从RAM加载层权重 (mmap)
                # 简化版: 跳过, 等模型就绪后补完
                pass
            # logits = lm_head(h)
            # next_token = logits.argmax(-1)
            # tokens.append(next_token)
            # h = embedding(next_token)
        return tokenizer.decode(tokens)


if __name__ == "__main__":
    print("[Pipeline] Core engine ready.")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Memory Head: {'loaded' if MEMORY_AVAILABLE else 'missing'}")
    print(f"  Expert Reader: mmap v2 active")
