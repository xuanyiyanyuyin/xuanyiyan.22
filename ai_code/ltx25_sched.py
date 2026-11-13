"""LTX-2.5 22B video-only DiT 手写引擎 — 从 0.9.8 ltx_sched 魔改
保留: 三层调度/offload/pread 基建
改写: 权重加载(单文件safetensors按块pread) + video-only forward
删除: audio/av_ca 相关
"""
import sys, io, os, struct, json, gc, threading, math, atexit, signal
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from concurrent.futures import ThreadPoolExecutor

DTYPE_MAP = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16,
             'I64': torch.int64, 'BOOL': torch.bool, 'I32': torch.int32}
DEV = 'cuda'
HEAD_DIM = 128; NUM_HEADS = 32; HIDDEN = 4096; EPS = 1e-6
# LTX-2.5 配置 (从config确认)
DIT_FILE = r'D:\ai\_ltx25_src\ltx-2.5-22b-dev-transformer-bf16_v2.safetensors'  # V2 干净, V1 含 NaN(block6/39 ff)
NUM_BLOCKS = 48
PATCHIFY_IN = 128          # latent channels
PATCHIFY_OUT = 4096        # hidden

# ═══ CUDA cleanup guard (同 0.9.8) ═══
def _cleanup_cuda():
    try:
        torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    except Exception: pass
atexit.register(_cleanup_cuda)
def _sigint_handler(sig, frame):
    _cleanup_cuda(); raise KeyboardInterrupt
try: signal.signal(signal.SIGINT, _sigint_handler)
except ValueError: pass

# ═══ 单文件 safetensors pread 访问层 ═══
_header_cache = None
def _load_header():
    global _header_cache
    if _header_cache is None:
        with open(DIT_FILE, 'rb') as f:
            hs = struct.unpack('<Q', f.read(8))[0]
            _header_cache = (8 + hs, json.loads(f.read(hs)))
    return _header_cache

_tls = threading.local()
def _get_fd():
    fd = getattr(_tls, 'fd', None)
    if fd is None:
        fd = open(DIT_FILE, 'rb'); _tls.fd = fd
    return fd

def pread_tensor(key):
    """按 key 从单文件 pread 一个 tensor"""
    base, header = _load_header()
    if key not in header:
        raise KeyError(f'未找到权重: {key}')
    meta = header[key]
    o0, o1 = meta['data_offsets']
    n = o1 - o0
    f = _get_fd()
    f.seek(base + o0)
    data = f.read(n)
    return torch.frombuffer(data, dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])

P = 'model.diffusion_model.'
def _k(name): return P + name

def read_block(lid):
    """按块读取 video-only 权重 (排除 audio/av_ca)"""
    prefix = P + f'transformer_blocks.{lid}.'
    base, header = _load_header()
    rels = []
    for k in header:
        if not k.startswith(prefix): continue
        rel = k[len(prefix):]
        if rel.startswith(('audio_', 'scale_shift_table_a2v_')): continue
        rels.append(rel)
    return {rel: pread_tensor(prefix + rel) for rel in rels}

def read_global():
    """顶层 video-only 权重 (once) — 注意: 不含 video_embeddings_connector (3.23GB, forward用不到, 省显存)"""
    out = {}
    base, header = _load_header()
    for k in header:
        if not k.startswith(P): continue
        rel = k[len(P):]
        if rel.startswith(('audio_', 'av_ca_')): continue
        if rel.split('.')[0] in ('patchify_proj', 'adaln_single', 'prompt_adaln_single',
                                 'scale_shift_table', 'proj_out'):
            out[rel] = pread_tensor(k)
    return out

# ═══ 基础运算 ═══
def rms_norm(x, w, eps=EPS):
    """和官方一致: F.rms_norm (bf16计算, 保精度对齐)"""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=w, eps=eps)

def linear(x, w, b=None):
    """⚠️ 把 x 转成权重的 dtype, 不是反过来 — 反之会在每次调用时把 bf16 权重复制成 fp32
    (FFN 权重 268MB/次 × 48 块 × 每步 → 就是 2K 慢 20 倍的根因, 2026-09-11 修)"""
    if x.dtype != w.dtype: x = x.to(w.dtype)
    if b is not None: return F.linear(x, w, b.to(w.dtype))
    return F.linear(x, w)

# ═══ LTX-2.5 video-only forward ═══
NUM_HEADS = 32; HEAD_DIM = HIDDEN // NUM_HEADS  # 4096/32 = 128

def ada_zero(x, scale, shift, eps=EPS):
    """AdaLN-Zero: rms_norm(x) * (1+scale) + shift
    scale/shift (B,D) → unsqueeze(1) 成 (B,1,D) 广播到 (B,T,D) (CFG B=2 两路)"""
    return rms_norm(x, None, eps) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def apply_rope_pe(x, pe, w=None):
    """官方 apply_split_rotary_emb (全5维运算): x(B,T,D) → (B,H,T,head_dim) → 交错旋转"""
    cos, sin = pe  # 各 (B, H, T, rope_dim=64)
    if cos.device != x.device:
        cos, sin = cos.to(x.device), sin.to(x.device)
    H = cos.shape[1]
    xh = x.unflatten(-1, (H, -1)).transpose(1, 2)  # (B,H,T,head_dim)
    sp = xh.unflatten(-1, (2, -1))  # (B,H,T,2,rope_dim)
    c = cos.unsqueeze(-2); s = sin.unsqueeze(-2)  # (B,H,T,1,rope_dim)
    output = sp * c  # (B,H,T,2,rope_dim)
    first = output[..., :1, :]; second = output[..., 1:, :]
    first_h = sp[..., 1:, :]; second_h = sp[..., :1, :]
    first.addcmul_(-s, first_h)
    second.addcmul_(s, second_h)
    out = output.flatten(-2)  # (B,H,T,head_dim)
    return out.transpose(1, 2).flatten(-2)  # (B,T,D)

# ═══ RoPE pe 生成 (官方 SPLIT) ═══
def _freq_grid(theta, max_pos_count, inner_dim):
    n_elem = 2 * max_pos_count
    indices = theta ** torch.linspace(0.0, 1.0, inner_dim // n_elem, dtype=torch.float32)
    return indices * math.pi / 2.0

def _fractional_positions(indices_grid, max_pos):
    # indices_grid (B, n_dims, T) 或 (B, n_dims, T, 2)
    n_dims = indices_grid.shape[1]
    if indices_grid.ndim == 4:
        indices_grid = (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    frac = []
    for d in range(n_dims):
        mp = max_pos[d]
        frac.append(indices_grid[:, d:d+1] / mp)
    return torch.cat(frac, dim=1)  # (B, n_dims, T)

def compute_pe(positions, inner_dim=HIDDEN, max_pos=(20, 2048, 2048), num_heads=NUM_HEADS, theta=10000.0):
    """positions (B,3,T,2) → (cos, sin) 各 (B,H,T,rope_dim=64)
    直接用官方 precompute_freqs_cis (保证正确, 独立重写留作后续)"""
    import sys as _s
    if not any('ltx-core' in p for p in _s.path):
        _s.path.insert(0, r'D:\ai\_ltx25_src\LTX-2-main\packages\ltx-core\src')
    from ltx_core.model.transformer.rope import precompute_freqs_cis, LTXRopeType
    return precompute_freqs_cis(positions, inner_dim, torch.bfloat16, theta=theta,
                                max_pos=list(max_pos), use_middle_indices_grid=True,
                                num_attention_heads=num_heads, rope_type=LTXRopeType.SPLIT)

def attention_forward(x, context, pe, w, is_self):
    """LTX-2.5 Attention: to_q/k/v + q_norm/k_norm(全4096) + RoPE + SDPA + gate + to_out"""
    B, T, D = x.shape
    ctx = x if is_self else context
    q = linear(x, w['to_q.weight'], w['to_q.bias'])
    k = linear(ctx, w['to_k.weight'], w['to_k.bias'])
    v = linear(ctx, w['to_v.weight'], w['to_v.bias'])
    q = rms_norm(q, w['q_norm.weight']); k = rms_norm(k, w['k_norm.weight'])
    if pe is not None and is_self:
        q = apply_rope_pe(q, pe, w); k = apply_rope_pe(k, pe, w)
    Sk = k.shape[1] if k.ndim == 3 else k.shape[1]
    qh = q.view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    kh = k.view(B, Sk, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    vh = v.view(B, Sk, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    # 统一 dtype: 官方 SDPA 要求 q/k/v 同 dtype (cross-attn 的 text_mod 可能引入 float)
    if kh.dtype != qh.dtype: kh = kh.to(qh.dtype)
    if vh.dtype != qh.dtype: vh = vh.to(qh.dtype)
    # 官方 PytorchAttention 用 sdpa_kernel(priority=[CUDNN>FLASH>EFFICIENT>MATH]) 包裹,
    # 不同后端在 bf16 下有 ~4e-3 数值差异, 48层串联放大 → 必须一致
    _prio = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    with sdpa_kernel(_prio, set_priority=True):
        out = F.scaled_dot_product_attention(qh, kh, vh)  # SDPA自带1/sqrt(d)
    out = out.transpose(1, 2)  # (B,T,32,128)
    if 'to_gate_logits.weight' in w:
        gate = 2.0 * torch.sigmoid(linear(x, w['to_gate_logits.weight'], w['to_gate_logits.bias']))
        out = out * gate[:, :, :, None]  # per-head gate
    out = out.reshape(B, T, D)
    return linear(out, w['to_out.0.weight'], w['to_out.0.bias'])

def ff_forward(x, w):
    h = F.gelu(linear(x, w['net.0.proj.weight'], w.get('net.0.proj.bias')), approximate='tanh')
    return linear(h, w['net.2.weight'], w.get('net.2.bias'))

def block_forward(x, text_emb, mods, pe, prompt_kv, mask, w):
    """video-only transformer block (LTX-2.5)"""
    sst = w['scale_shift_table']  # (9,4096)
    # ⚠️ sst 存的是 fp32, 直接相加会把整块计算污染成 fp32 (2K 慢 20 倍根因) → 强制回算力 dtype
    block_mods = (mods + sst.unsqueeze(0)).to(x.dtype)  # (B,9,4096) [0,1,2]sa [3,4,5]ffn [6,7,8]cross
    m = block_mods
    w1 = {k[6:]: v for k, v in w.items() if k.startswith('attn1.')}
    w2 = {k[6:]: v for k, v in w.items() if k.startswith('attn2.')}
    ffw = {k[3:]: v for k, v in w.items() if k.startswith('ff.')}
    # self-attention [0,1,2]
    x_norm = ada_zero(x, m[:, 1], m[:, 0])       # scale_sa, shift_sa
    x = x + attention_forward(x_norm, None, pe, w1, is_self=True) * m[:, 2].unsqueeze(1)
    # cross-attention [6,7,8] + prompt KV mod
    x_norm = rms_norm(x, None)
    q_shift, q_scale, q_gate = m[:, 6].unsqueeze(1), m[:, 7].unsqueeze(1), m[:, 8].unsqueeze(1)
    attn_in = x_norm * (1 + q_scale) + q_shift
    text_mod = text_emb * (1 + prompt_kv[1].unsqueeze(1)) + prompt_kv[0].unsqueeze(1)  # scale_kv, shift_kv
    if mask is not None:
        am = ((1 - mask.float()) * -10000.0).unsqueeze(1).unsqueeze(1)
    else:
        am = None
    x = x + attention_forward(attn_in, text_mod, None, w2, is_self=False) * q_gate
    # FFN [3,4,5]
    x_norm = ada_zero(x, m[:, 4], m[:, 3])
    x = x + ff_forward(x_norm, ffw) * m[:, 5].unsqueeze(1)
    return x

# ═══ 顶层 forward ═══
TIMESTEP_SCALE_MULTIPLIER = 1000  # 源码默认 (timestep 0-1000 × 1000)

def _adaln_single(timestep, gw):
    """adaln_single: Timesteps(256, flip_sin_to_cos=True) → linear1 → silu → linear2 → linear(9×)
    源码: 先 × timestep_scale_multiplier(1000) 再进 HF Timesteps"""
    dev = gw['adaln_single.linear.weight'].device
    t = (timestep.float() * TIMESTEP_SCALE_MULTIPLIER).to(dev)
    half = 128
    # HF Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half  # /128 不是 /127
    emb = torch.exp(exponent)
    args = t[:, None] * emb[None, :]
    t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B,256)
    t_emb = torch.cat([t_emb[:, half:], t_emb[:, :half]], dim=-1)  # flip_sin_to_cos: cos在前
    # 官方 cast 到 hidden_dtype (bf16) 再进 linear
    t_emb = t_emb.to(gw['adaln_single.emb.timestep_embedder.linear_1.weight'].dtype)
    # 官方: linear_1 → SiLU → linear_2(embedded) → SiLU → linear(9×)
    h = F.silu(linear(t_emb, gw['adaln_single.emb.timestep_embedder.linear_1.weight'], gw['adaln_single.emb.timestep_embedder.linear_1.bias']))
    ada = linear(h, gw['adaln_single.emb.timestep_embedder.linear_2.weight'], gw['adaln_single.emb.timestep_embedder.linear_2.bias'])
    mods = linear(F.silu(ada), gw['adaln_single.linear.weight'], gw['adaln_single.linear.bias'])
    # 返回 (mods, embedded_timestep): 官方 AdaLayerNormSingle 返回 (linear(silu(emb)), emb)
    # embedded_timestep = linear_2 输出, 用于输出层 scale_shift_table 动态调制
    return mods.view(t.shape[0], 9, HIDDEN), ada.view(t.shape[0], 1, HIDDEN)

def _prompt_kv(sigma, gw, w):
    """prompt_adaln_single(sigma) + block prompt_scale_shift_table → (scale_kv, shift_kv)
    官方: prompt_adaln_single 应用在 modality.sigma (timestep式嵌入), 不是 text_emb"""
    dev = gw['prompt_adaln_single.linear.weight'].device
    s = (sigma.float() * TIMESTEP_SCALE_MULTIPLIER).to(dev)
    # prompt_adaln_single: timestep_embedder + linear(2×)
    half = 128
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=dev) / half
    emb = torch.exp(exponent)
    args = s[:, None] * emb[None, :]
    t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    t_emb = torch.cat([t_emb[:, half:], t_emb[:, :half]], dim=-1)
    t_emb = t_emb.to(gw['prompt_adaln_single.emb.timestep_embedder.linear_1.weight'].dtype)
    h = F.silu(linear(t_emb, gw['prompt_adaln_single.emb.timestep_embedder.linear_1.weight'], gw['prompt_adaln_single.emb.timestep_embedder.linear_1.bias']))
    ada = linear(h, gw['prompt_adaln_single.emb.timestep_embedder.linear_2.weight'], gw['prompt_adaln_single.emb.timestep_embedder.linear_2.bias'])
    out = linear(F.silu(ada), gw['prompt_adaln_single.linear.weight'], gw['prompt_adaln_single.linear.bias'])
    # 官方: kv_modulation = prompt_scale_shift_table[None,None] + prompt_timestep.reshape(B,1,2,D)
    #       unbind(dim=2) → row0=shift_kv, row1=scale_kv; context*(1+scale_kv)+shift_kv
    # 手写修正: 先取 shift=row0, scale=row1, 再分别加 pss[0]/pss[1]
    shift, scale = out.view(sigma.shape[0], 2, HIDDEN).unbind(dim=1)  # (B,4096) each
    pss = w.get('prompt_scale_shift_table')
    if pss is not None:
        shift = shift + pss[0].unsqueeze(0)
        scale = scale + pss[1].unsqueeze(0)
    # 转 bf16 (官方 context 是 bf16, 调制系数须同 dtype, 否则 text_mod 变 float 导致 SDPA dtype 不匹配)
    shift = shift.to(torch.bfloat16); scale = scale.to(torch.bfloat16)
    return shift, scale

# block 权重预取: 后台线程 pread + H2D，与主线程 block 计算重叠（隐藏磁盘 I/O）
_block_prefetch_pool = ThreadPoolExecutor(max_workers=2)

def _load_block_gpu(lid):
    """read_block(lid) 并 H2D — 供预取线程调用"""
    return {k: v.to(DEV) for k, v in read_block(lid).items()}

def transformer_forward_sched(hidden_states, text_emb, timestep, mask,
                              num_frames, height, width, pe=None, dtype=torch.bfloat16):
    """LTX-2.5 video-only DiT forward: patchify → 48块 → proj_out
    text_emb: connector 输出 (B,S_text,4096) — 由 Gemma4+connector 前置模块产生
    预取: block lid+1 的权重在 block lid 计算时后台加载，隐藏磁盘 I/O"""
    gw = read_global()
    gwgpu = {k: v.to(DEV) for k, v in gw.items()}
    try:
        B, S, C = hidden_states.shape
        x = linear(hidden_states.to(DEV), gwgpu['patchify_proj.weight'], gwgpu['patchify_proj.bias'])  # (B,S,4096)
        text_emb = text_emb.to(DEV)
        mods, embedded_timestep = _adaln_single(timestep, gwgpu)  # (B,9,4096), (B,1,4096)
        next_fut = None
        for lid in range(NUM_BLOCKS):
            if next_fut is not None:
                w = next_fut.result()
            else:
                w = _load_block_gpu(0)
            if lid + 1 < NUM_BLOCKS:
                next_fut = _block_prefetch_pool.submit(_load_block_gpu, lid + 1)
            prompt_kv = _prompt_kv(timestep, gwgpu, w)
            x = block_forward(x, text_emb, mods, pe, prompt_kv, mask, w)
            del w
        # 输出 (官方 _process_output): scale_shift_table[None,None] + embedded_timestep[:,:,None]
        # → (B,1,2,D); shift=[0], scale=[1]; norm_out=LayerNorm(elementwise_affine=False)
        sst = gwgpu['scale_shift_table'].to(x.dtype)  # (2,4096) ⚠️ fp32→bf16 防污染
        sv = sst[None, None] + embedded_timestep[:, :, None].to(x.dtype)  # (B,1,2,4096)
        shift, scale = sv[:, :, 0], sv[:, :, 1]  # 各 (B,1,4096)
        x = F.layer_norm(x, (HIDDEN,), weight=None, bias=None, eps=EPS)
        x = x * (1 + scale) + shift
        out = linear(x, gwgpu['proj_out.weight'], gwgpu['proj_out.bias'])
        return out
    finally:
        del gwgpu; gc.collect()

def cleanup():
    torch.cuda.empty_cache(); gc.collect()

if __name__ == '__main__':
    g = read_global()
    print('global 加载:', {k: tuple(v.shape) for k, v in g.items() if k in ('patchify_proj.weight', 'proj_out.weight', 'scale_shift_table')})
    b0 = read_block(0)
    print(f'block0: {len(b0)} tensors')
    print('  attn1.to_q:', tuple(b0['attn1.to_q.weight'].shape))
    print('  ff.net.0.proj:', tuple(b0['ff.net.0.proj.weight'].shape))
    print('  scale_shift_table:', tuple(b0['scale_shift_table'].shape))
