import sys, time, torch
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
t0=time.perf_counter()
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载 {time.perf_counter()-t0:.0f}s', flush=True)
# 手动分步: tokenizer
lm=pipe.text_encoder.model.language_model
print('lm OK, layers', len(lm.layers), flush=True)
# tokenize
prompts=['a cute orange cat']
sys_msg=pipe.system_message
cleaned=[p.replace('[IMG]','') for p in prompts]
messages=[[{'role':'system','content':sys_msg},{'role':'user','content':[{'type':'text','text':p}]}] for p in cleaned]
inp=pipe.tokenizer.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
print('tokenize OK', inp['input_ids'].shape, flush=True)
