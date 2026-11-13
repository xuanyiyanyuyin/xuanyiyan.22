import sys, time, torch
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
t0=time.perf_counter()
pipe=Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
lm=pipe.text_encoder.model.language_model
prompts=['a cute orange cat']; sys_msg=pipe.system_message
cleaned=[p.replace('[IMG]','') for p in prompts]
messages=[[{'role':'system','content':sys_msg},{'role':'user','content':[{'type':'text','text':p}]}] for p in cleaned]
inp=pipe.tokenizer.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
input_ids=inp['input_ids'].to('cuda'); attn=inp['attention_mask'].to('cuda')
lm.embed_tokens.to('cuda'); lm.rotary_emb.to('cuda')
h=lm.embed_tokens(input_ids)
pos=torch.arange(512,device='cuda').unsqueeze(0)
pe=lm.rotary_emb(h,position_ids=pos)
from transformers.models.mistral.modeling_mistral import create_causal_mask
cm=create_causal_mask(config=lm.config,inputs_embeds=h,attention_mask=attn,past_key_values=None,position_ids=pos)
# 不加empty_cache, 只 .to('cpu') 逐层
for i in range(40):
    t1=time.perf_counter()
    lm.layers[i].to('cuda')
    h=lm.layers[i](hidden_states=h,attention_mask=cm,position_ids=pos,past_key_values=None,use_cache=False,position_embeddings=pe)
    lm.layers[i].to('cpu')
    if i in (0,1,9,10,19,20,29,30,39):
        print(f'layer{i} {time.perf_counter()-t1:.1f}s cum={time.perf_counter()-t0:.0f}s', flush=True)
print(f'全部完成 {time.perf_counter()-t0:.0f}s', flush=True)
