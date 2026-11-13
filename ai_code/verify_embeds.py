import sys, torch, time
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
from transformers import AutoProcessor
t0=time.perf_counter()
tok = AutoProcessor.from_pretrained(f2.MODEL + '/tokenizer', local_files_only=True)
te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
print(f'pread {time.perf_counter()-t0:.0f}s', flush=True)
# 官方 encode_prompt 逻辑 (用 pread 的 te 替换)
# 手动: 官方 _get_mistral_3_small_prompt_embeds 用 te+tok
from transformers.models.mistral.modeling_mistral import create_causal_mask
prompt='a cute orange cat'
sys_msg='You are a helpful assistant'
messages=[[{'role':'system','content':sys_msg},{'role':'user','content':[{'type':'text','text':prompt}]}]]
inp=tok.apply_chat_template(messages,add_generation_prompt=False,tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
input_ids=inp['input_ids']; attn=inp['attention_mask']
print('tokenize OK', flush=True)
# 官方在CPU/GPU? encode用官方完整 forward (te 全模型, 非逐层)
# 用 te(input_ids) 官方前向 (CPU, 45G慢但官方参照)
print('官方完整 te.forward (CPU参照)...', flush=True)
te.to('cpu')
with torch.inference_mode():
    out=te(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    # 官方取层10/20/30
    hs=[out.hidden_states[k] for k in (10,20,30)]
    st=torch.stack(hs,dim=1)
    bs,nc,sl,hd=st.shape
    official_embeds=st.permute(0,2,1,3).reshape(bs,sl,nc*hd).bfloat16()
print(f'官方embeds {time.perf_counter()-t0:.0f}s shape={tuple(official_embeds.shape)}', flush=True)
# 对比我缓存的
ck=torch.load(r'D:\ai\_cached_flux2_embeds_test.pt', map_location='cpu', weights_only=True)
my=ck['embeds']
diff=(official_embeds-my).abs()
print(f'官方 vs 我的: max_diff={diff.max():.4f} mean={diff.mean():.5f}', flush=True)
torch.save({'official':official_embeds.cpu(),'mine':my.cpu()}, r'D:\ai\_embeds_cmp.pt')
print('VERIFY_DONE', flush=True)
