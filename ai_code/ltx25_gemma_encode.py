"""LTX-2.5 Gemma4 文本编码前端 — prompt → text_emb (4096)
用法: python ltx25_gemma_encode.py "prompt"
产出: text_emb (1, S, 4096) bf16

⚠️ 本机 safetensors mmap 会段错误 (os error 1455 页面文件太小) —
   所有权重读取必须用 pread (struct header + seek data_offsets), 绕开 mmap。
"""
import sys, torch, time, json, struct
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
import psutil

GEMMA_FILE = r'D:\ai\_ltx25_src\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors'
DIT_FILE = r'D:\ai\_ltx25_src\ltx-2.5-22b-dev-transformer-bf16_v2.safetensors'  # V2 干净

DTYPE_MAP = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16, 'U8': torch.uint8}


def load_header(path):
    f = open(path, 'rb'); hs = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(hs)); f.close()
    return 8 + hs, h


def pread_tensor_from(path, header, base, key):
    """从单文件 pread 一个 tensor (绕开 mmap 段错误)"""
    m = header[key]
    o0, o1 = m['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + o0); data = f.read(o1 - o0)
    dt = DTYPE_MAP.get(m['dtype'])
    if dt is None:
        raise ValueError(f'未知 dtype {m["dtype"]} for {key}')
    return torch.frombuffer(data, dtype=dt).reshape(m['shape'])


def pread_tokenizer_json(path):
    """从 single-file pread tokenizer_json (U8 bytes)"""
    base, hdr = load_header(path)
    m = hdr['tokenizer_json']
    o0, o1 = m['data_offsets']
    with open(path, 'rb') as f:
        f.seek(base + o0); data = f.read(o1 - o0)
    # U8 tensor → bytes (tokenizer.json 原始字节)
    import numpy as np
    return np.frombuffer(data, dtype=np.uint8).tobytes()


def main(prompt):
    t0 = time.time()
    # 1. tokenizer (pread 读 tokenizer_json, 绕开 GemmaAssets 段错误)
    from tokenizers import Tokenizer
    tok_bytes = pread_tokenizer_json(GEMMA_FILE)
    tok = Tokenizer.from_str(tok_bytes.decode('utf-8'))
    print(f'[gemma] tokenizer pread OK {time.time()-t0:.0f}s', flush=True)

    # 2. Gemma4 三层缓存逐层编码 (照 122B 模式: 磁盘pread → 官方层模块, 峰值2.5GB, 绕开 from_pretrained 段错误)
    from ltx25_gemma4_engine import encode_gemma
    enc = tok.encode(prompt)
    MAXLEN = 1024   # 官方 max_sequence_length (LTX2Pipeline 默认)
    pad_len = (MAXLEN - len(enc.ids) % MAXLEN) % MAXLEN
    ids_list = enc.ids + [tok.token_to_id('<pad>')] * pad_len
    ids = torch.tensor([ids_list])
    mask_full = torch.zeros_like(ids)
    mask_full[0, :len(enc.ids)] = 1
    hs = encode_gemma(ids, mask_full, device='cpu')  # tuple of 49 hidden_states (1,S,3840)
    print(f'[gemma] encode {time.time()-t0:.0f}s tokens={ids.shape[1]} layers={len(hs)}', flush=True)

    # 4. FeatureExtractorV2: norm_and_concat + video_aggregate_embed (pread 读)
    from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2
    base, gemma_meta = load_header(GEMMA_FILE)
    agg_w = pread_tensor_from(GEMMA_FILE, gemma_meta, base,
                              'text_embedding_projection.video_aggregate_embed.weight')
    agg_b = pread_tensor_from(GEMMA_FILE, gemma_meta, base,
                              'text_embedding_projection.video_aggregate_embed.bias')
    agg = torch.nn.Linear(agg_w.shape[1], agg_w.shape[0]).to(torch.bfloat16)
    agg.weight.data.copy_(agg_w); agg.bias.data.copy_(agg_b)
    print(f'[gemma] aggregate embed pread OK {time.time()-t0:.0f}s', flush=True)

    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnectorConfigurator
    fx = FeatureExtractorV2(video_aggregate_embed=agg, embedding_dim=4096)
    dit_base, dit_meta = load_header(DIT_FILE)
    meta = {'config': json.loads(dit_meta['__metadata__']['config'])}
    conn = Embeddings1DConnectorConfigurator.from_metadata(meta)
    # pread 加载 connector 权重 (从 DiT 文件)
    P = 'model.diffusion_model.video_embeddings_connector.'
    conn_sd = {}
    for k in dit_meta:
        if not k.startswith(P): continue
        conn_sd[k[len(P):]] = pread_tensor_from(DIT_FILE, dit_meta, dit_base, k)
    conn.load_state_dict(conn_sd)
    conn = conn.to(torch.bfloat16)
    print(f'[gemma] connector pread OK {time.time()-t0:.0f}s', flush=True)

    proc = EmbeddingsProcessor(feature_extractor=fx, video_connector=conn)
    res = proc.process_hidden_states(hs, mask_full, padding_side='left')
    video_encoded = res.video_encoding
    print(f'[gemma] video_encoded: {tuple(video_encoded.shape)} {time.time()-t0:.0f}s', flush=True)

    # 5. 输出 text_emb (文件名可由 argv[2] 指定, 默认 text_emb.pt)
    text_emb = video_encoded
    print(f'[gemma] text_emb: {tuple(text_emb.shape)} std={text_emb.float().std().item():.4f} {time.time()-t0:.0f}s', flush=True)
    import os
    os.makedirs(r'D:\ai\output\ab_test', exist_ok=True)
    out_name = sys.argv[2] if len(sys.argv) > 2 else 'text_emb.pt'
    torch.save(text_emb.float().cpu(), r'D:\ai\output\ab_test' + '\\' + out_name)
    print(f'[gemma] 已存 {out_name}  RAM={psutil.Process().memory_info().rss/1e9:.1f}GB', flush=True)
    return text_emb


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'a cute orange cat sitting on a wooden table'
    main(p)
