# -*- coding: utf-8 -*-
import sys, time, os, torch
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE']='1'
from diffusers import Flux2Pipeline
import flux2_3tier as f2
def mem(t):
    print(f'  [{t}] GPU={torch.cuda.memory_allocated()/1024**3:.2f}G', flush=True)
t0=time.perf_counter()
print('加载 pipeline...', flush=True)
pipe=Flux2Pipeline.from_pretrained(r'D:\ai\models\FLUX.2-dev', torch_dtype=torch.bfloat16, local_files_only=True)
print(f'加载完 {time.perf_counter()-t0:.0f}s', flush=True)
# 释放 text_encoder (验证阶段不用)
del pipe.text_encoder; torch.cuda.empty_cache()
# patch transformer
r=f2._patch_f2(pipe.transformer, resident_d=0)
print(f'patch完 常驻{r}块', flush=True)
# 准备 latent + 假 embeds (短prompt)
nch=pipe.transformer.config.in_channels//4
lat,latent_ids=pipe.prepare_latents(1,nch,256,256,torch.bfloat16,'cuda',None,None)
print(f'latent {tuple(lat.shape)} latent_ids {tuple(latent_ids.shape)}', flush=True)
# 假 text embeds: shape 从 encode_prompt 推 (batch, seq, 3*hidden)
# 用随机 (1, 16, 4) txt_ids, prompt_embeds (1, 20, dim)
num_txt=16
dim=lat.shape[-1]*4 if False else 4096  # 探测
# 实际得看 transformer 期望 encoder_hidden_states 维度
enc=torch.randn(1,num_txt,3072,dtype=torch.bfloat16,device='cuda')
txt_ids=torch.arange(num_txt,device='cuda').unsqueeze(0).expand(1,num_txt,4).bfloat16() if False else torch.zeros(1,num_txt,4,device='cuda',dtype=torch.bfloat16)
guid=torch.full([1],3.5,device='cuda',dtype=torch.float32)
ts=torch.full([1],50,device='cuda',dtype=torch.float32)
print('跑 transformer 1步...', flush=True)
try:
    out=pipe.transformer(hidden_states=lat, timestep=ts/1000, guidance=guid,
                         encoder_hidden_states=enc, txt_ids=txt_ids, img_ids=latent_ids,
                         joint_attention_kwargs=None, return_dict=False)
    print(f'✅ transformer 跑通! out {tuple(out[0].shape)} {time.perf_counter()-t0:.0f}s', flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print('❌ transformer 报错:', type(e).__name__, str(e)[:300], flush=True)
mem('done')
