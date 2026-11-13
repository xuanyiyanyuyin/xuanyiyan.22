import sys, time, torch, psutil
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
from transformers import AutoProcessor
import faulthandler; faulthandler.enable()
t0=time.perf_counter()
tok = AutoProcessor.from_pretrained(f2.MODEL + '/tokenizer', local_files_only=True)
print('tokenizer OK', flush=True)
te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
print(f'pread {time.perf_counter()-t0:.0f}s', flush=True)
lm = te.model.language_model
print('开始 GPU 逐层编码...', flush=True)
with torch.inference_mode():
    prompts=['a cute orange cat']
    messages=[[{'role':'system','content':'You are a helpful assistant'},{'role':'user','content':[{'type':'text','text':p}]}] for p in prompts]
    inp=tok.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
    input_ids=inp['input_ids'].to('cuda'); attn=inp['attention_mask'].to('cuda')
    lm.embed_tokens.to('cuda'); lm.rotary_emb.to('cuda'); lm.norm.to('cuda')
    h=lm.embed_tokens(input_ids)
    pos=torch.arange(512,device='cuda').unsqueeze(0)
    pe=lm.rotary_emb(h,position_ids=pos)
    from transformers.models.mistral.modeling_mistral import create_causal_mask
    cm=create_causal_mask(config=lm.config,inputs_embeds=h,attention_mask=attn,past_key_values=None,position_ids=pos)
    hs={}
    for i in range(40):
        t1=time.perf_counter()
        lm.layers[i].to('cuda')
        h2=lm.layers[i](hidden_states=h,attention_mask=cm,position_ids=pos,past_key_values=None,use_cache=False,position_embeddings=pe)
        del h; h=h2; del h2
        if (i+1) in (10,20,30): hs[i+1]=h.detach().clone()   # 官方 hidden_states[k]=layers[k-1]; i从0起 => 存 k=i+1
        lm.layers[i].to('cpu')
        torch.cuda.empty_cache()
        if i%5==0: print(f'layer{i} {time.perf_counter()-t1:.1f}s cum={time.perf_counter()-t0:.0f}s', flush=True)
    out=torch.stack([hs[k] for k in (10,20,30)],dim=1).float()
    bs,nc,sl,hd=out.shape
    pe2=out.permute(0,2,1,3).reshape(bs,sl,nc*hd).bfloat16()
    torch.save({'embeds':pe2.cpu()}, r'D:\ai\_cached_flux2_embeds_test.pt')
    print(f'编码完成 {time.perf_counter()-t0:.0f}s embeds={tuple(pe2.shape)}', flush=True)
print('ENCODE_OK', flush=True)
