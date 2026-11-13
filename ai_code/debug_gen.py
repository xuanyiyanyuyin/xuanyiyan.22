"""修复版生成测试 — 逐步调试"""
import torch, gc
import sys; sys.path.insert(0, r"D:\ai")
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

prompt = "你好，请问"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to("cuda")
print(f"Input: {tokenizer.decode(input_ids[0])} -> {input_ids}")

past_kv = DynamicCache()

# Prefill
hidden = mod(input_ids, None, past_key_values=past_kv, position_ids=None)
logits = torch.matmul(hidden[:, -1:, :].float(), mod.embed_tokens.weight.t().float())
next_tok = logits.argmax(dim=-1)
print(f"Step 0 (prefill): '{tokenizer.decode(next_tok[0])}' (id={next_tok.item()})")

generated = torch.cat([input_ids, next_tok], dim=-1)

# Decode loop
for step in range(1, 20):
    new_tok = generated[:, -1:]
    pos = torch.tensor([[generated.shape[1] - 1]], device="cuda")
    hidden = mod(new_tok, None, past_key_values=past_kv, position_ids=pos)
    logits = torch.matmul(hidden[:, -1:, :].float(), mod.embed_tokens.weight.t().float())
    next_tok = logits.argmax(dim=-1)
    generated = torch.cat([generated, next_tok], dim=-1)
    print(f"Step {step}: '{tokenizer.decode(next_tok[0])}' (id={next_tok.item()})")
    if next_tok.item() == tokenizer.eos_token_id:
        break

print(f"\nFinal: {tokenizer.decode(generated[0])}")
print(mgr.stats())
mgr.cleanup()
