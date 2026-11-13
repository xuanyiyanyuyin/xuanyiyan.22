"""
VL-30B 文本层单层校准 — 官方第一层 vs 手写第一层
纯文本输入, 对比 attention + MoE 输出
"""
import torch, torch.nn.functional as F, sys, io, json, struct, os
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

# ═══ 配置 ═══
cfg=json.load(open(f'{MODEL}/config.json'))
tc=cfg['text_config']
H=tc['hidden_size']; NH=tc['num_attention_heads']; NKV=tc['num_key_value_heads']
HD=tc['head_dim']; EP=tc['rms_norm_eps']
N_EXPERTS=128; TOP_K=8; EXP_INTER=768
ROPE_DIM=64; MR_SECTION=tc['rope_scaling']['mrope_section']  # [24,20,20]
ROPE_THETA=tc['rope_theta']

# ═══ 加载第一层权重 ═══
DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64}
def load_keys(path, wanted):
    out={}
    with open(path,'rb') as f:
        hs=struct.unpack('<Q',f.read(8))[0]
        hdr=json.loads(f.read(hs)); base=8+hs
        for k in wanted:
            m=hdr[k]; o0,o1=m['data_offsets']
            f.seek(base+o0); buf=f.read(o1-o0)
            out[k]=torch.frombuffer(buf,dtype=DTYPE_MAP[m['dtype']]).reshape(m['shape'])
    return out

idx=json.load(open(f'{MODEL}/model.safetensors.index.json'))
wm=idx['weight_map']
# 第一层非专家 + 专家
layer0_keys={}
for k,v in wm.items():
    if 'layers.0.' in k and 'expert' not in k:
        layer0_keys.setdefault(v,[]).append(k)
lw={}
for shard,names in layer0_keys.items():
    tensors=load_keys(f'{MODEL}/{shard}', names)
    for k,v in tensors.items():
        lw[k.replace('model.language_model.','').replace('layers.0.','')]=v
    del tensors
print(f'[calib] layer0 keys: {len(lw)}')

# 专家
ek=load_keys(f'{MODEL}/model-00001-of-00013.safetensors',
    ['model.language_model.layers.0.mlp.experts.gate_up_proj','model.language_model.layers.0.mlp.experts.down_proj'])
g=ek['model.language_model.layers.0.mlp.experts.gate_up_proj'][0].to(DEV)
d=ek['model.language_model.layers.0.mlp.experts.down_proj'][0].to(DEV)  # expert 0
print(f'[calib] expert0 gate_up {g.shape} down {d.shape}')

# ═══ 官方第一层 ═══
from transformers import AutoConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextAttention, Qwen3VLMoeTextRotaryEmbedding)

full_cfg=AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tc_obj=full_cfg.text_config
tc_obj._attn_implementation='eager'

S=12
x=torch.randn(1,S,H,device=DEV,dtype=torch.bfloat16)
# 官方 attention
attn=Qwen3VLMoeTextAttention(tc_obj, layer_idx=0).to(DEV).to(torch.bfloat16).eval()
# 装权重
sd_attn={k.replace('self_attn.','').replace('layers.0.',''):v for k,v in lw.items() if 'self_attn' in k}
with torch.no_grad():
    for k,v in attn.state_dict().items():
        if k in sd_attn: v.copy_(sd_attn[k].to(v.dtype))
# rope
rope=Qwen3VLMoeTextRotaryEmbedding(tc_obj).to(DEV)
pids=torch.arange(S,device=DEV).view(1,1,-1).expand(3,1,-1)
cos,sin=rope(x, pids)
with torch.no_grad():
    ref_attn,_=attn(x,(cos,sin),attention_mask=None,past_key_values=None)

# 手写 attention
def w(k): return lw.get(k)
q=w('self_attn.q_proj.weight').to(DEV); k=w('self_attn.k_proj.weight').to(DEV)
v=w('self_attn.v_proj.weight').to(DEV); o=w('self_attn.o_proj.weight').to(DEV)
qn=w('self_attn.q_norm.weight').to(DEV); kn=w('self_attn.k_norm.weight').to(DEV)
Q=(x@q.T).view(1,S,NH,HD).float()
if qn is not None: Q=Q*torch.rsqrt(Q.pow(2).mean(-1,keepdim=True)+EP)*qn.float()
qa=Q.transpose(1,2).to(torch.bfloat16)
K=(x@k.T).view(1,S,NKV,HD).float()
if kn is not None: K=K*torch.rsqrt(K.pow(2).mean(-1,keepdim=True)+EP)*kn.float()
K=K.transpose(1,2).to(torch.bfloat16); Vm=(x@v.T).view(1,S,NKV,HD).transpose(1,2).to(torch.bfloat16)
# mrope 应用: 用官方cos/sin (T维=cos[0])
def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)
# cos/sin shape: [3, bs, seq, 64], 取T维(纯文本)
# cos/sin [1,12,128] = [bs,seq,head_dim] → [bs,1,seq,head_dim]
c=cos.unsqueeze(1); s=sin.unsqueeze(1)
c_q=c.expand(-1,NH,-1,-1); s_q=s.expand(-1,NH,-1,-1)
qa_rot=qa*c_q+rotate_half(qa)*s_q
c_k=c.expand(-1,NKV,-1,-1); s_k=s.expand(-1,NKV,-1,-1)
K_rot=K*c_k+rotate_half(K)*s_k
K_rot=K_rot.repeat_interleave(NH//NKV,1); Vm=Vm.repeat_interleave(NH//NKV,1)
att=F.scaled_dot_product_attention(qa_rot,K_rot,Vm,is_causal=False)
mine_attn=att.transpose(1,2).contiguous().view(1,S,NH*HD).to(torch.bfloat16)@o.T

# 对比
diff=(ref_attn.float()-mine_attn.float()).abs()
rel=diff.norm()/ref_attn.float().norm()
print(f'[calib] attn max_abs={diff.max().item():.4f} rel={rel.item():.6f}')
print(f'[calib] ref[:2]: {ref_attn[0,0,:3].tolist()}')
print(f'[calib] mine[:2]: {mine_attn[0,0,:3].tolist()}')
