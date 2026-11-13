"""
VL-30B 完整prefill校准 — 官方模型(纯文本) vs 手写引擎
小输入, 跑一次prefill, 逐层对比hidden state
"""
import torch, torch.nn.functional as F, sys, io, json, struct, os
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

cfg=json.load(open(f'{MODEL}/config.json'))
tc=cfg['text_config']
H=tc['hidden_size']; EP=tc['rms_norm_eps']; N_LAYERS=tc['num_hidden_layers']

# ═══ 手写引擎加载 ═══
from vl_scorer import VLScorer
scorer=VLScorer()
print(f'[calib] handwritten engine ready (vision={type(scorer.vis_model).__name__})')

# ═══ 构造纯文本输入 (无图) ═══
from transformers import AutoProcessor
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
msgs=[{'role':'user','content':'你好, 请介绍一下你自己'}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=proc(text=[text], return_tensors='pt')
input_ids=inp['input_ids'].to(DEV)
print(f'[calib] input_ids {tuple(input_ids.shape)}: {input_ids[0][:10].tolist()}')

# ═══ 手写 prefill (无视觉) ═══
h=F.embedding(input_ids, scorer.emb)
S=h.shape[1]
cur_pos=S
cache={}
mine_outs=[]
self=scorer
self._prefetch_layer(0)
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
        flat=hn2.to(torch.bfloat16).reshape(-1,H)
        out,_=self._moe_forward(flat,gpu,lid)
        h=(h+out.to(torch.bfloat16).reshape_as(h))
        del gpu
        if lid in (0,1,3,11,23,47):
            mine_outs.append((lid,h.clone()))
print(f'[calib] handwritten prefill done, {S} tokens')

# 最终norm + logits
hf=h.float()
hnorm=hf*torch.rsqrt(hf.pow(2).mean(-1,keepdim=True)+EP)*self.final_norm.float()
logits=(hnorm[:,-1:,:].to(torch.bfloat16)@self.lm_head.T).float()
mine_tok=int(logits[0,-1].argmax())
print(f'[calib] mine next token: {mine_tok} = {repr(proc.decode([mine_tok]))}')
# 保存 logits 供官方对比 (释放前)
torch.save(logits.float().cpu(), r'D:\ai\offload\mine_logits.pt')
print('[calib] saved mine logits')

# ═══ 官方 prefill ═══
print('[calib] loading official model (may be slow)...')
from transformers import Qwen3VLMoeForConditionalGeneration
# 释放手写引擎GPU占用
del scorer
import gc; gc.collect(); torch.cuda.empty_cache()
print('[calib] handwritten released')
official=Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map='auto',
    max_memory={0:'6GiB','cpu':'22GiB'},
    trust_remote_code=True, local_files_only=True,
    offload_folder=r'D:\ai\offload')
official.eval()
with torch.no_grad():
    out=official(input_ids=input_ids, use_cache=False)
ref_logits=out.logits[0,-1]
ref_tok=int(ref_logits.argmax())
logits=torch.load(r'D:\ai\offload\mine_logits.pt').to(DEV)
print(f'[calib] official next token: {ref_tok} = {repr(proc.decode([ref_tok]))}')
print(f'[calib] MATCH: {mine_tok==ref_tok}')

# logits对比
diff=(logits.float()-ref_logits.float()).abs()
rel=diff.norm()/ref_logits.float().norm()
print(f'[calib] logits max_abs={diff.max().item():.4f} rel={rel.item():.6f}')
top5_mine=logits[0].topk(5).indices.tolist()
top5_ref=ref_logits.topk(5).indices.tolist()
print(f'[calib] mine top5: {[proc.decode([t]) for t in top5_mine]}')
print(f'[calib] ref  top5: {[proc.decode([t]) for t in top5_ref]}')
