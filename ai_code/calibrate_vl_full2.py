"""
VL-30B 完整视觉prefill校准 v2 — 官方 vs 手写, next token对比
"""
import torch, torch.nn.functional as F, sys, io, json, os
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

# ═══ 输入 ═══
from transformers import AutoProcessor
from PIL import Image
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
img=Image.new('RGB',(256,256),color=(200,60,60))
msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':'这是什么?'}]}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=proc(text=[text], images=[img], return_tensors='pt')
input_ids=inp['input_ids'].to(DEV)
pv=inp['pixel_values']; thw=inp['image_grid_thw']
print(f'[calib] ids {tuple(input_ids.shape)} pv {tuple(pv.shape)}')

# ═══ 手写完整prefill ═══
from vl_scorer import VLScorer, EP, IMG_TOKEN
scorer=VLScorer()
self=scorer
with torch.inference_mode():
    vis=self._vision_forward(pv, thw)
h=F.embedding(input_ids, self.emb)
ids=input_ids[0]; vis_mask=(ids==IMG_TOKEN); vis_idx=torch.where(vis_mask)[0]
for i,pos in enumerate(vis_idx[:vis.shape[1]]):
    h[0,pos]=vis[0,i]
# pos_ids 构建 (官方 get_rope_index)
S=h.shape[1]
t_grid,gh,gw=thw[0].tolist()
sm=2; n_vis=vis_idx.numel()
lg_t=t_grid; lg_h=gh//sm; lg_w=gw//sm
v_t=torch.zeros(n_vis,dtype=torch.long,device=DEV)
v_h=torch.zeros(n_vis,dtype=torch.long,device=DEV)
v_w=torch.zeros(n_vis,dtype=torch.long,device=DEV)
t_i=torch.arange(lg_t).view(-1,1).expand(-1,lg_h*lg_w).flatten()
h_i=torch.arange(lg_h).view(1,-1,1).expand(lg_t,-1,lg_w).flatten()
w_i=torch.arange(lg_w).view(1,1,-1).expand(lg_t,lg_h,-1).flatten()
n_grid_tok=lg_t*lg_h*lg_w
v_t[:n_grid_tok]=t_i; v_h[:n_grid_tok]=h_i; v_w[:n_grid_tok]=w_i
text_start=int(max(v_h.max().item(),v_w.max().item())+1)
text_len=S-n_vis
tt=torch.arange(text_len,device=DEV).view(1,-1).expand(3,-1)+text_start
t_pos=torch.zeros(S,dtype=torch.long,device=DEV); h_pos=torch.zeros_like(t_pos); w_pos=torch.zeros_like(t_pos)
t_pos[vis_mask]=v_t; h_pos[vis_mask]=v_h; w_pos[vis_mask]=v_w
t_pos[~vis_mask]=tt[0]; h_pos[~vis_mask]=tt[1]; w_pos[~vis_mask]=tt[2]
self.pos_ids=torch.stack([t_pos,h_pos,w_pos])

# prefill
cache={}; cur_pos=S
self._prefetch_layer(0)
N_LAYERS=self.layer_ram.__len__()
with torch.no_grad():
    for lid in range(N_LAYERS):
        gpu=self._get_layer(lid)
        if lid+1<N_LAYERS: self._prefetch_layer(lid+1)
        inn=gpu.get('input_layernorm.weight')
        hn=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if inn is not None: hn=hn*inn.float()
        hn=hn.to(torch.bfloat16)
        a_out=self._full_attn(hn,gpu,lid,True,cache,cur_pos)
        h=h+a_out
        pn=gpu.get('post_attention_layernorm.weight')
        hn2=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
        if pn is not None: hn2=hn2*pn.float()
        flat=hn2.to(torch.bfloat16).reshape(-1,2048)
        out,_=self._moe_forward(flat,gpu,lid)
        h=(h+out.to(torch.bfloat16).reshape_as(h))
        del gpu
torch.save(h.float().cpu(), r'D:\ai\offload\mine_h0.pt')
hnorm=h.float()*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)*self.final_norm.float()
logits=(hnorm[:,-1:,:].to(torch.bfloat16)@self.lm_head.T).float()[0]
print(f'[calib] mine next: {int(logits.argmax())} = {repr(proc.decode([int(logits.argmax())]))}')
torch.save(logits.float().cpu(), r'D:\ai\offload\mine_vis_logits.pt')
print('[calib] saved')
del scorer; import gc; gc.collect(); torch.cuda.empty_cache()

# ═══ 官方 ═══
print('[calib] loading official...')
from transformers import Qwen3VLMoeForConditionalGeneration
official=Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0:'6GiB','cpu':'22GiB'}, trust_remote_code=True,
    local_files_only=True, offload_folder=r'D:\ai\offload')
official.eval()
with torch.no_grad():
    out=official(input_ids=input_ids, pixel_values=pv.to(DEV).to(torch.bfloat16),
                 image_grid_thw=thw.to(DEV), use_cache=False, output_hidden_states=True)
# 对比第0层 hidden
ref_h0=out.hidden_states[0]  # [1,77,2048] 视觉注入后的emb
print(f'[calib] official h0 shape {tuple(ref_h0.shape)}')
torch.save(ref_h0.float().cpu(), r'D:\ai\offload\ref_h0.pt')
ref_logits=out.logits[0,-1]
print(f'[calib] official next: {int(ref_logits.argmax())} = {repr(proc.decode([int(ref_logits.argmax())]))}')
mine_logits=torch.load(r'D:\ai\offload\mine_vis_logits.pt').to(DEV)
diff=(mine_logits-ref_logits.float()).abs()
rel=diff.norm()/ref_logits.float().norm()
print(f'[calib] logits max_abs={diff.max().item():.4f} rel={rel.item():.6f}')
print(f'[calib] MATCH={int(mine_logits.argmax())==int(ref_logits.argmax())}')
# 对比第0层输入 (视觉注入后)
mine_h0=torch.load(r'D:\ai\offload\mine_h0.pt')
ref_h0=torch.load(r'D:\ai\offload\ref_h0.pt')
d0=(mine_h0-ref_h0).abs()
r0=d0.norm()/ref_h0.norm()
print(f'[calib] h0 max_abs={d0.max().item():.4f} rel={r0.item():.6f}')
# 分token对比
for tok in [0,1,2,10,50,76]:
    dk=(mine_h0[0,tok]-ref_h0[0,tok]).abs().max().item()
    print(f'[calib]   tok{tok} max_abs={dk:.4f}')
