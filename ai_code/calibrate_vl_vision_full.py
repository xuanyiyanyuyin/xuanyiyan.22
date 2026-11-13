"""
VL-30B 完整视觉prefill校准 — 官方 vs 手写, 对比中间hidden
"""
import torch, torch.nn.functional as F, sys, io, json, os
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'
H=2048; EP=1e-6

# ═══ 构造输入 ═══
from transformers import AutoProcessor
from PIL import Image
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
img=Image.new('RGB',(256,256),color=(200,60,60))
msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':'这是什么?'}]}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=proc(text=[text], images=[img], return_tensors='pt')
input_ids=inp['input_ids'].to(DEV)
pv=inp['pixel_values']; thw=inp['image_grid_thw']
print(f'[calib] input_ids {tuple(input_ids.shape)}, pv {tuple(pv.shape)}, thw {thw}')

# ═══ 手写引擎 ═══
from vl_scorer import VLScorer
scorer=VLScorer()
self=scorer
with torch.inference_mode():
    vis=self._vision_forward(pv, thw)
h=F.embedding(input_ids, self.emb)
ids=input_ids[0]
vis_mask=(ids==self.IMG_TOKEN)
vis_idx=torch.where(vis_mask)[0]
for i,pos in enumerate(vis_idx[:vis.shape[1]]):
    h[0,pos]=vis[0,i]
# 构建pos_ids (复用score_image逻辑)
S=h.shape[1]
import sys as _s
# 直接调 score_image 的 pos_ids 构建
# 简化: 手动构建
t_grid,gh,gw=thw[0].tolist()
sm=2; per_grid=4; n_vis=vis_idx.numel(); n_grid=n_vis//per_grid
v_t=torch.zeros(n_vis,dtype=torch.long,device=DEV)
v_h=torch.zeros(n_vis,dtype=torch.long,device=DEV)
v_w=torch.zeros(n_vis,dtype=torch.long,device=DEV)
for gi in range(n_grid):
    hi=(gi//gw)*sm; wi=(gi%gw)*sm; base=gi*per_grid
    for dy in range(sm):
        for dx in range(sm):
            p=base+dy*sm+dx
            v_h[p]=hi+dy; v_w[p]=wi+dx
mv=max(v_t.max().item(),v_h.max().item(),v_w.max().item())
ts=mv+1
tt=torch.arange(ts,ts+(S-n_vis),device=DEV)
t_pos=torch.zeros(S,dtype=torch.long,device=DEV); h_pos=torch.zeros_like(t_pos); w_pos=torch.zeros_like(t_pos)
t_pos[vis_mask]=v_t; h_pos[vis_mask]=v_h; w_pos[vis_mask]=v_w
t_pos[~vis_mask]=tt; h_pos[~vis_mask]=tt; w_pos[~vis_mask]=tt
self.pos_ids=torch.stack([t_pos,h_pos,w_pos])

# 手写 prefill 第0层
self._prefetch_layer(0)
gpu=self._get_layer(0)
inn=gpu.get('input_layernorm.weight')
hn=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
if inn is not None: hn=hn*inn.float()
hn=hn.to(torch.bfloat16)
cache={}
a_out=self._full_attn(hn,gpu,0,True,cache,S)
h_after=h+a_out
# 保存手写中间值
torch.save(h_after.float().cpu(), r'D:\ai\offload\mine_layer0.pt')
torch.save(h.float().cpu(), r'D:\ai\offload\mine_input_emb.pt')
print('[calib] handwritten layer0 done, saving...')
del scorer; import gc; gc.collect(); torch.cuda.empty_cache()

# ═══ 官方模型 ═══
print('[calib] loading official...')
from transformers import Qwen3VLMoeForConditionalGeneration
official=Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0:'6GiB','cpu':'22GiB'}, trust_remote_code=True,
    local_files_only=True, offload_folder=r'D:\ai\offload')
official.eval()
with torch.no_grad():
    out=official(input_ids=input_ids, pixel_values=pv.to(DEV).to(torch.bfloat16),
                 image_grid_thw=thw.to(DEV), use_cache=False)

# 对比最终logits
mine_emb=torch.load(r'D:\ai\offload\mine_input_emb.pt')
mine_l0=torch.load(r'D:\ai\offload\mine_layer0.pt')
# 官方第0层hidden不容易直接拿, 用logits对比
ref_logits=out.logits[0,-1]
print(f'[calib] official logits[:5]: {ref_logits[:5].tolist()}')
# 手写最终logits (跑完整prefill太重, 先对比第0层)
# 官方第0层: 用 hook 或直接比较 embeddings
# 简化: 只报官方的next token
print(f'[calib] official next: {proc.decode([int(ref_logits.argmax())])}')
