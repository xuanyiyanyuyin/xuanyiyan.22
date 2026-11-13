import sys, torch
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
lm = te.model.language_model
# 检查哪些参数还是 meta
meta_cnt=0; total=0
for name,p in lm.named_parameters():
    total+=1
    if p.is_meta: 
        meta_cnt+=1
        if meta_cnt<=5: print('META:', name)
print(f'语言模型 meta 参数: {meta_cnt}/{total}')
# embed 检查
print('embed_tokens is_meta:', lm.embed_tokens.weight.is_meta)
print('norm is_meta:', lm.norm.weight.is_meta)
# 检查非 language_model 部分 (vision/projector) 是否 meta
for n,p in te.named_parameters():
    if p.is_meta:
        print('META(te):', n)
