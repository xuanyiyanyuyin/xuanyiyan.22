import os, sys, torch
sys.path.insert(0, r"D:\ai\py_h3")
import accelerate
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
H3 = r"D:\ai\models\MiniMax-H3\Ref2VA"
with accelerate.init_empty_weights():
    pipe = MiniMaxH3ModularPipeline.from_pretrained(H3, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
print("=== pipeline 组件（含类型）===")
for n in dir(pipe):
    if n.startswith("_"): continue
    try:
        v = getattr(pipe, n)
    except Exception: continue
    t = type(v).__name__
    if t in ("ModularPipelineBlocks","SequentialPipelineBlocks","ConditionalPipelineBlocks"): continue
    if any(k in n.lower() for k in ("transformer","vae","encoder","tokenizer","processor","scheduler","blocks")):
        print("  %-28s %s" % (n, t))
print("=== 可用方法（前 25）===")
ms = [m for m in dir(pipe) if not m.startswith("_") and callable(getattr(pipe, m, None))]
print("  ", ms[:25])
import gc; gc.collect(); os._exit(0)