"""对比: prefill(一次性) vs decode(逐token+cache)"""
import torch, sys
sys.path.insert(0, r"D:\ai")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from storage_manager import StorageManager
from modified_qwen3 import ModifiedQwen3Model

MODEL_PATH = "D:/ai/models/qwen3-0.6b"

orig_cpu = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
mgr = StorageManager(device="cuda", ram_device="cpu")
mgr.register_model(orig_cpu)
mod = ModifiedQwen3Model(orig_cpu.model, mgr)

# 3个token一次性 (prefill)
tokens = torch.tensor([[108386, 37945, 56007]], device="cuda")  # 你好，请问
print("=== Prefill (3 tokens at once) ===")
hidden_prefill = mod(tokens, None, past_key_values=None, position_ids=None)
logits_prefill = torch.matmul(hidden_prefill[:, -1:, :].float(), mod.embed_tokens.weight.t().float())
tok_prefill = logits_prefill.argmax(dim=-1)
print(f"  Last token logits argmax: {tok_prefill.item()} '{tokenizer.decode(tok_prefill[0])}'")
print(f"  Top-5: {[(tokenizer.decode([t]), logits_prefill[0,0,t].item()) for t in logits_prefill[0,0].topk(5).indices]}")

# 3个token逐个 (decode with cache)
print("\n=== Decode (1 token at a time with cache) ===")
past_kv = DynamicCache()
hidden_state = None

for i, tok_id in enumerate([108386, 37945, 56007]):
    t = torch.tensor([[tok_id]], device="cuda")
    pos = torch.tensor([[i]], device="cuda")
    hidden_state = mod(t, None, past_key_values=past_kv, position_ids=pos)
    print(f"  Token {i} ({tokenizer.decode([tok_id])}): hidden mean={hidden_state.float().mean():.4f}")

logits_decode = torch.matmul(hidden_state[:, -1:, :].float(), mod.embed_tokens.weight.t().float())
tok_decode = logits_decode.argmax(dim=-1)
print(f"  Last token logits argmax: {tok_decode.item()} '{tokenizer.decode(tok_decode[0])}'")
print(f"  Top-5: {[(tokenizer.decode([t]), logits_decode[0,0,t].item()) for t in logits_decode[0,0].topk(5).indices]}")

# 关键对比
print(f"\n=== COMPARISON ===")
print(f"Prefill vs Decode logits match? {(logits_prefill - logits_decode).abs().max():.6f}")
print(f"Prefill token: {tok_prefill.item()} Decode token: {tok_decode.item()} {'MATCH' if tok_prefill.item() == tok_decode.item() else 'MISMATCH!!!'}")

mgr.cleanup()
