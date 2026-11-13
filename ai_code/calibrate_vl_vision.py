"""
VL-30B Vision Encoder 锚点校准
官方vision model vs 手写_vision_forward, 逐层对比
"""
import torch, sys, io, json, struct, os
sys.path.insert(0, r'D:\ai')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
DEV='cuda'
MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

# ═══ 从checkpoint提取visual权重 ═══
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
vis_keys={}
for name,shard in idx['weight_map'].items():
    if 'visual' in name:
        vis_keys.setdefault(shard,[]).append(name)
print(f'[calib] loading vision weights from {len(vis_keys)} shards...')
vis_w={}
for shard,names in vis_keys.items():
    tensors=load_keys(f'{MODEL}/{shard}', names)
    for k,v in tensors.items():
        vis_w[k.replace('model.visual.','')]=v  # 去掉前缀
    del tensors
print(f'[calib] {len(vis_w)} vision tensors')

# ═══ 加载官方vision model (CPU, 手动装权重) ═══
from transformers import Qwen3VLMoeVisionModel
model=Qwen3VLMoeVisionModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
model.eval()
# 手动覆盖权重 (官方key `blocks.0.x` 对应 checkpoint `blocks.0.x`)
loaded=0
for k,v in model.state_dict().items():
    if k in vis_w:
        with torch.no_grad():
            model.state_dict()[k].copy_(vis_w[k].to(model.state_dict()[k].dtype))
        loaded+=1
print(f'[calib] loaded {loaded} weights into official model')
model=model.to(DEV)

# ═══ 准备输入: 小图 ═══
from PIL import Image
from transformers import AutoProcessor
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
img=Image.new('RGB',(512,512),color=(200,60,60))
msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':'x'}]}]
text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=proc(text=[text], images=[img], return_tensors='pt')
pv=inp['pixel_values']; thw=inp['image_grid_thw']
print(f'[calib] pixel_values {pv.shape}, grid_thw {thw}')

# ═══ 官方 forward ═══
with torch.inference_mode():
    ref=model(pv.to(DEV), grid_thw=thw.to(DEV))
ref_out=ref if isinstance(ref,torch.Tensor) else ref[0]
print(f'[calib] official vision out: {tuple(ref_out.shape)}')

# ═══ 手写 forward ═══
from vl_scorer import VLScorer
scorer=VLScorer()  # 会加载vision权重到self.vis_w (键无前缀, 匹配)
with torch.inference_mode():
    mine=scorer._vision_forward(pv, thw)
print(f'[calib] handwritten vision out: {tuple(mine.shape)}')

# ═══ 对比 ═══
# 对齐形状: mine [1,N,2048] → [N,2048]
mine2 = mine[0] if mine.dim()==3 else mine
ref2 = ref_out[0] if ref_out.dim()==3 else ref_out
if mine2.shape==ref2.shape:
    diff=(ref2.float()-mine2.float()).abs()
    rel=diff.norm()/ref2.float().norm()
    print(f'[calib] max_abs={diff.max().item():.4f} rel_err={rel.item():.6f}')
    print(f'[calib] ref sample: {ref2[0,:5].tolist()}')
    print(f'[calib] mine sample: {mine2[0,:5].tolist()}')
    # cosine sim
    cos=torch.nn.functional.cosine_similarity(ref2.float().flatten(), mine2.float().flatten(), dim=0)
    print(f'[calib] cosine_sim={cos.item():.6f}')
else:
    print(f'[calib] SHAPE MISMATCH: ref {ref2.shape} vs mine {mine2.shape}')
