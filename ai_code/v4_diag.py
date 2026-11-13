"""V4-Flash 乱码定位实验: 增量decode(C) vs 全量重算(B) 的 logits 偏差
若 C 偏离 B → 病在增量/缓存状态路径(compressor/indexer state), 与 llama.cpp 已知 bug 同类
若 B 本身就怪 → 病在单遍 forward(权重转换/专家数学)
"""
import sys, torch, time
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import v4flash_sched as vs
from transformers import AutoTokenizer

eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)

# DSML 工具调用 prompt (官方 encoding)
try:
    from encoding_dsv4 import encode_messages
    dsml = None
    for tm in ('false', 'disabled', 'off', 'enabled'):
        try:
            dsml = encode_messages([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer via the calculator tool."},
            ], thinking_mode=tm)
            print(f'[prompt] encode_messages ok thinking_mode={tm}, {len(dsml)} chars', flush=True)
            break
        except Exception as e:
            print(f'[prompt] thinking_mode={tm} fail: {e}', flush=True)
    if dsml is None:
        dsml = '<｜begin▁of▁sentence｜>User: tool call\n\nAssistant:'
else:
    dsml = '<｜begin▁of▁sentence｜>User: tool call\n\nAssistant:'

ids_d = tok(dsml, return_tensors='pt').input_ids.to('cuda')
ids_p = tok('User: What is the capital of France?\n\nAssistant:', return_tensors='pt').input_ids.to('cuda')
print(f'[ids] plain={ids_p.shape[1]} dsml={ids_d.shape[1]}', flush=True)

def diag(ids, label):
    with torch.inference_mode():
        logits1, cache = eng.forward(ids)
    t1 = logits1[:, -1].argmax(-1, keepdim=True)
    ids2 = torch.cat([ids, t1], 1)
    with torch.inference_mode():
        logitsB, _ = eng.forward(ids2)
        logitsC, _ = eng.forward(t1, past_key_values=cache)
    b = logitsB[0, -1].float(); c = logitsC[0, -1].float()
    cos = torch.nn.functional.cosine_similarity(b, c, dim=0).item()
    md = (b - c).abs().max().item()
    topB = b.topk(5).indices.tolist(); topC = c.topk(5).indices.tolist()
    print(f'[{label}] cos(B,C)={cos:.4f} maxdiff={md:.2f}', flush=True)
    print(f'  B top5: {tok.decode(topB)!r}', flush=True)
    print(f'  C top5: {tok.decode(topC)!r}', flush=True)
    print(f'  token1: {tok.decode([t1.item()])!r}', flush=True)

diag(ids_p, 'plain')
diag(ids_d, 'dsml')
print('DIAG DONE', flush=True)
