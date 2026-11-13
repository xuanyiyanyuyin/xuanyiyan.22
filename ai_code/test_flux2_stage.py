import torch, sys, time, os, gc
sys.path.insert(0, r'D:\ai')
MODEL=r'D:\ai\models\FLUX.2-dev'
def mem(tag):
    import psutil
    print(f'  [{tag}] RAM avail={psutil.virtual_memory().available/1024**3:.1f}G GPU={torch.cuda.memory_allocated()/1024**3:.1f}G', flush=True)

from diffusers import Flux2Pipeline
print('加载(但先只处理 text_encoder)...', flush=True)
# 用 device_map 让 text_encoder 不整块驻留? 先试 from_pretrained 观察
pipe=Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
mem('加载后')
# 把 transformer 挪到 meta/或删掉只留 text_encoder? 
# 关键测试: text_encoder 单独编码, 看内存峰值
pipe.text_encoder = pipe.text_encoder.to('cuda' if torch.cuda.is_available() else 'cpu') if False else pipe.text_encoder
# 只保留 text_encoder, transformer 先不管
del pipe.transformer, pipe.vae
gc.collect()
import psutil
mem('删transformer后')
# 编码
inputs=pipe.tokenizer('a cute orange cat', return_tensors='pt')
if torch.cuda.is_available():
    inputs={k:v.to('cuda') for k,v in inputs.items()}
    pipe.text_encoder.to('cuda')
with torch.no_grad():
    out=pipe.text_encoder(**inputs)
mem('编码后')
print('编码输出 shape:', out.last_hidden_state.shape if hasattr(out,'last_hidden_state') else out.shape, flush=True)
# 编码完释放 text_encoder
pipe.text_encoder.to('cpu'); del pipe.text_encoder; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
mem('释放text_encoder后')
print('OK 分阶段可行?', flush=True)
