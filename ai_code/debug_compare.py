"""对比原版模型 vs 魔改版模型, 定位 forward 正确性 bug"""
import torch
import sys; sys.path.insert(0, r"D:\ai")
from transformers import AutoModelForCausalLM, AutoTokenizer
from storage_manager import StorageManager
from modified_qwen3 import ModifiedQwen3Model

MODEL_PATH = "D:/ai/models/qwen3-0.6b"

# 加载原版模型 (GPU)
print("[1] 原版模型...")
orig = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载魔改版
print("[2] 魔改版模型...")
orig_cpu = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
mgr = StorageManager(device="cuda", ram_device="cpu")
mgr.register_model(orig_cpu)
mod = ModifiedQwen3Model(orig_cpu.model, mgr)

# 测试输入
prompt = "你好"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs.input_ids
print(f"  input_ids: {input_ids.shape}, {input_ids}")

# ── 原版: 看每层输出 ──
print("\n[3] 原版 forward (逐层)...")
orig_hidden = orig.model.embed_tokens(input_ids)
position_ids = torch.arange(0, input_ids.shape[1], device="cuda").unsqueeze(0)
pos_emb = orig.model.rotary_emb(orig_hidden, position_ids)

for i in range(3):  # 只看前3层
    layer_out = orig.model.layers[i](orig_hidden, None, position_ids, None, False, pos_emb)
    print(f"  Layer {i}: mean={layer_out.float().mean():.6f} std={layer_out.float().std():.6f} min={layer_out.float().min():.4f} max={layer_out.float().max():.4f}")
    orig_hidden = layer_out

# ── 魔改版: 看每层输出 ──
print("\n[4] 魔改版 forward (逐层)...")
mod_hidden = mod.embed_tokens(input_ids)
pos_emb2 = mod.rotary_emb(mod_hidden, position_ids)

for i in range(3):
    layer = mod.layers[i]
    layer_out = layer(mod_hidden, pos_emb2, None, None)
    print(f"  Layer {i}: mean={layer_out.float().mean():.6f} std={layer_out.float().std():.6f} min={layer_out.float().min():.4f} max={layer_out.float().max():.4f}")

    # 比较差异
    diff = (layer_out.float() - orig.model.layers[i](mod_hidden, None, position_ids, None, False, pos_emb2).float()).abs()
    print(f"         vs_orig: max_diff={diff.max():.6f} mean_diff={diff.mean():.8f}")
    mod_hidden = layer_out

mgr.cleanup()
print("\nDone")
