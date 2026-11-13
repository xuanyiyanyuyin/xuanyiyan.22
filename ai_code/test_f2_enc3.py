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
print('tokenize OK', flush=True)
# embed
lm.embed_tokens.to('cuda')
h=lm.embed_tokens(input_ids)
print('embed OK', h.shape, flush=True)
# rotary
seq=input_ids.shape[1]
pos=torch.arange(seq,device='cuda').unsqueeze(0)
lm.rotary_emb.to('cuda')
pe=lm.rotary_emb(h, position_ids=pos)
print('rotary OK', flush=True)
# causal mask
from transformers.models.mistral.modeling_mistral import create_causal_mask
try:
    cm=create_causal_mask(config=lm.config, inputs_embeds=h, attention_mask=attn, past_key_values=None, position_ids=pos)
    print('causal_mask OK', type(cm), flush=True)
except Exception as e:
    print('causal_mask FAIL:', type(e).__name__, e, flush=True)
    import traceback; traceback.print_exc()
# 跑一层
lm.layers[0].to('cuda')
try:
    o=lm.layers[0](hidden_states=h, attention_mask=cm, position_ids=pos, past_key_values=None, use_cache=False, position_embeddings=pe)
    print('layer0 OK', type(o), flush=True)
except Exception as e:
    print('layer0 FAIL:', type(e).__name__, e, flush=True)
    import traceback; traceback.print_exc()
