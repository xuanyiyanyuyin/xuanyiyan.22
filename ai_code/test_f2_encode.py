import sys, time, torch, psutil
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
import os
t0=time.perf_counter()
# 只加载 text_encoder + 编码 (不加载整个 pipeline的transformer, 省时间)
from diffusers import Flux2Pipeline
# 需要 pipe 提供 tokenizer/system_message/_prepare_text_ids, 但我们手动构造一个轻pipe
# 直接手动: pread text_encoder + 手动 encode (encode_prompt_streaming 需要 pipe)
# 用 from_pretrained 只拿 tokenizer 等轻组件? 不, from_pretrained会mmap整个
# 改为: 只 pread text_encoder, 用轻量方式拿 tokenizer
from transformers import AutoProcessor
tok = AutoProcessor.from_pretrained(f2.MODEL + '/tokenizer', local_files_only=True)
print('tokenizer OK', flush=True)
print('pread text_encoder...', flush=True)
te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
print(f'pread完成 {time.perf_counter()-t0:.0f}s', flush=True)
# 手动编码 (不走 pipe.encode_prompt, 直接用 te + tok)
from transformers.models.mistral.modeling_mistral import create_causal_mask
with torch.inference_mode():
    prompts=['a cute orange cat']; 
    messages=[[{'role':'system','content':'You are a helpful assistant'},{'role':'user','content':[{'type':'text','text':p}]}] for p in prompts]
    inp=tok.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
    input_ids=inp['input_ids'].to('cuda'); attn=inp['attention_mask'].to('cuda')
    lm=te.model.language_model
    lm.embed_tokens.to('cuda'); lm.rotary_emb.to('cuda'); lm.norm.to('cuda')
    h=lm.embed_tokens(input_ids)
    pos=torch.arange(512,device='cuda').unsqueeze(0)
    pe=lm.rotary_emb(h,position_ids=pos)
    cm=create_causal_mask(config=lm.config,inputs_embeds=h,attention_mask=attn,past_key_values=None,position_ids=pos)
    hs={}
    for i in range(40):
        t1=time.perf_counter()
        lm.layers[i].to('cuda')
        h2=lm.layers[i](hidden_states=h,attention_mask=cm,position_ids=pos,past_key_values=None,use_cache=False,position_embeddings=pe)
        del h; h=h2; del h2
        if i in (10,20,30): hs[i]=h.detach().clone()
        lm.layers[i].to('cpu')
        torch.cuda.empty_cache()
        if i%5==0: print(f'layer{i} {time.perf_counter()-t1:.1f}s', flush=True)
    # 拼接
    out=torch.stack([hs[k] for k in (10,20,30)],dim=1).float()
    bs,nc,sl,hd=out.shape
    pe2=out.permute(0,2,1,3).reshape(bs,sl,nc*hd).bfloat16()
    torch.save({'embeds':pe2.cpu()}, r'D:\ai\_cached_flux2_embeds_test.pt')
    print(f'编码完成 {time.perf_counter()-t0:.0f}s embeds={tuple(pe2.shape)}', flush=True)
print('ENCODE_OK', flush=True)
