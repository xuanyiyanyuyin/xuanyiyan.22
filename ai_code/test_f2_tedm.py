import sys, time, torch
sys.path.insert(0, r'D:\ai')
import psutil
t0=time.perf_counter()
# 只加载 text_encoder 用 device_map=auto(自动 offload)
from transformers import AutoModelForCausalLM, AutoProcessor
print('尝试 transformers 加载 text_encoder (device_map=auto)...', flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(
        r'D:\ai\models\FLUX.2-dev\text_encoder', device_map='auto', torch_dtype=torch.bfloat16,
        local_files_only=True)
    print(f'加载OK, device_map 分布: {model.hf_device_map}', flush=True)
    print(f'总参 {sum(p.numel() for p in model.parameters())/1e9:.1f}B', flush=True)
    # 编码测试
    proc = AutoProcessor.from_pretrained(r'D:\ai\models\FLUX.2-dev\text_encoder', local_files_only=True)
    inp = proc('a cute cat', return_tensors='pt')
    with torch.no_grad():
        out = model(**inp)
    print('编码输出:', out.logits.shape if hasattr(out,'logits') else out.last_hidden_state.shape, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
print(f'done {time.perf_counter()-t0:.0f}s', flush=True)
