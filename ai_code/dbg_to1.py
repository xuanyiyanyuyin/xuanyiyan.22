import sys, time, torch, faulthandler
faulthandler.enable()
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
lm=pipe.text_encoder.model.language_model
print('单独搬 layer1 到GPU...', flush=True)
lm.layers[1].to('cuda')
torch.cuda.synchronize()
print('layer1.to(cuda) OK', flush=True)
