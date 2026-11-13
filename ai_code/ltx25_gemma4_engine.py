"""Gemma4 文本编码 — 严格三层缓存 (照 122B test_122b_fast.py 模式)
磁盘 pread → 逐层 → 官方 Gemma4TextDecoderLayer forward → hidden_states 累积
峰值内存: embed 2GB + 单层 0.45GB + hidden 累积 24MB ≈ 2.5GB
"""
import sys, json, struct, time, math
import torch
import torch.nn.functional as F

GEMMA_DIR = r'D:\ai\gemma4_text'
GEMMA_FILE = GEMMA_DIR + r'\model.safetensors'
DTYPE_MAP = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16, 'U8': torch.uint8}

# 架构参数
HIDDEN = 3840
NUM_LAYERS = 48


def _load_header():
    f = open(GEMMA_FILE, 'rb')
    hs = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hs))
    f.close()
    return 8 + hs, hdr


def pread_tensor(base, hdr, key):
    m = hdr[key]
    o0, o1 = m['data_offsets']
    with open(GEMMA_FILE, 'rb') as f:
        f.seek(base + o0)
        data = f.read(o1 - o0)
    return torch.frombuffer(data, dtype=DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone()


def load_embed(base, hdr):
    """embed_tokens + final norm (全程保留, RAM)"""
    gw = {}
    gw['embed_tokens'] = pread_tensor(base, hdr, 'model.embed_tokens.weight').to(torch.bfloat16)
    gw['norm'] = pread_tensor(base, hdr, 'model.norm.weight').to(torch.bfloat16)
    return gw


def encode_gemma(input_ids, attention_mask, device='cpu'):
    """三层缓存逐层 forward, 返回全部层 hidden_states (含 embedding 层)
    - 磁盘: pread 每层权重
    - RAM: embed 常驻 + 当前层权重
    - 计算: 官方 Gemma4TextDecoderLayer, 算完释放该层
    """
    import psutil
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding
    t0 = time.time()

    # config
    cfg = AutoConfig.from_pretrained(GEMMA_DIR, local_files_only=True)
    tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    layer_types = tc.layer_types  # 48 层

    # 磁盘→RAM: embed 常驻
    base, hdr = _load_header()
    gw = load_embed(base, hdr)
    embed_scale = math.sqrt(HIDDEN)

    B, S = input_ids.shape
    device_t = torch.device(device)
    h = F.embedding(input_ids.to(torch.long), gw['embed_tokens'].to(device_t))
    h = h * embed_scale
    hidden_states = [h.clone()]

    position_ids = torch.arange(S).unsqueeze(0).to(device_t)
    rotary = Gemma4TextRotaryEmbedding(tc)
    shared_kv_states = {}  # 跨层共享 KV (full 层的 KV 给后续 sliding 复用)

    for lid in range(NUM_LAYERS):
        lt = layer_types[lid] if lid < len(layer_types) else 'sliding_attention'
        # 磁盘: pread 该层权重 (0.45GB)
        prefix = f'model.layers.{lid}.'
        layer_sd = {}
        for k in hdr:
            if k.startswith(prefix) and k != '__metadata__':
                layer_sd[k[len(prefix):]] = pread_tensor(base, hdr, k).to(torch.bfloat16)
        # 官方模块 (每层重建, 因 sliding/full head_dim 不同; 用完释放)
        layer_module = Gemma4TextDecoderLayer(tc, lid).to(torch.bfloat16).eval()
        layer_module.load_state_dict(layer_sd, strict=True)
        # 官方 RoPE
        cos, sin = rotary(h, position_ids=position_ids, layer_type=lt)
        with torch.inference_mode():
            h = layer_module(h, position_embeddings=(cos, sin), shared_kv_states=shared_kv_states)
        hidden_states.append(h.clone())
        del layer_sd, layer_module
        import gc; gc.collect()
        if lid % 8 == 7:
            print(f'  layer {lid:2d} done RAM={psutil.Process().memory_info().rss/1e9:.1f}GB ({time.time()-t0:.0f}s)', flush=True)

    # final norm
    h = F.rms_norm(h, (HIDDEN,), gw['norm'].to(device_t), tc.rms_norm_eps)
    hidden_states[-1] = h
    print(f'[gemma] 48层完成 RAM={psutil.Process().memory_info().rss/1e9:.1f}GB ({time.time()-t0:.0f}s)', flush=True)
    return tuple(hidden_states)


if __name__ == '__main__':
    import torch
    print('Gemma4 三层缓存编码引擎 (由 ltx25_gemma_encode.py 调用)')
