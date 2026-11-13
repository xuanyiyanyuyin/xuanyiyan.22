"""V4-Flash 引擎 v3 — 三层缓存 (照 122B test_122b_fast.py / qwen_repo 基建)
- 非专家权重: 逐层 pread → RAM → 官方层 forward (单层放 GPU, 算完释放)
- 专家: PreadExpertReader 按需 pread 活性 top-6 + RAM LRU (OrderedDict), 不打包 3D
- forward 数学: attention/mHC/gate/shared 用官方 DeepseekV4DecoderLayer; 专家 FP8 GEMM 手写
  (参考官方 FP8Experts.forward 数学), 全程 fp32 激活
峰值: embed 1GB + 单层非专家 ~0.3GB + 活性专家 ~0.2GB ≈ 1.5GB GPU / <2GB CPU
"""
import os, sys, json, struct, time, re, gc, threading, copy, ctypes
from concurrent.futures import ThreadPoolExecutor
# 低优先级运行 (照 122B), 避免抢占前台资源
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

# FP8 内核 (finegrained-fp8 v4, 本地缓存直接加载)
_KERNEL_DIR = Path(r'C:/Users/Jinqi Wei/.cache/huggingface/hub/kernels--kernels-community--finegrained-fp8/snapshots/7cdb05d472d6c954c7d03182ed836ebfd4610df0/build/torch-cuda')

def _load_fp8_kernel():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_fp8_kernel', _KERNEL_DIR / '__init__.py', submodule_search_locations=[str(_KERNEL_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_fp8_kernel'] = mod
    spec.loader.exec_module(mod)
    return mod

FP8_KERNEL = _load_fp8_kernel()
import transformers.integrations.finegrained_fp8 as _ff8
_ff8.lazy_load_kernel = lambda *a, **k: FP8_KERNEL

MODEL_DIR = Path(r'D:\ai_v4flash')
DEVICE = 'cuda'
BLOCK_SIZE = 128
DTYPE_MAP = {'F8_E4M3': torch.float8_e4m3fn, 'F8_E8M0': torch.float8_e8m0fnu, 'BF16': torch.bfloat16,
             'F32': torch.float32, 'F16': torch.float16, 'I8': torch.int8, 'I64': torch.int64}


# ═══ pread 基础 (绕开 mmap) ═══
def _load_disk_index():
    idx = json.load(open(MODEL_DIR / 'model.safetensors.index.json'))
    return idx['weight_map']

DISK = _load_disk_index()

def pread(key: str) -> torch.Tensor:
    shard = DISK[key]
    with open(MODEL_DIR / shard, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hs))
        base = 8 + hs
        m = hdr[key]
        o0, o1 = m['data_offsets']
        f.seek(base + o0)
        data = f.read(o1 - o0)
        return torch.frombuffer(data, dtype=DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone()


# ═══ key 转换 (非专家; 照 transformers conversion_mapping.py deepseek_v4) ═══
RENAME_RULES = [
    (r'\.scale$', r'.weight_scale_inv'),
    (r'\.attn\.', r'.self_attn.'),
    (r'\.ffn\.', r'.mlp.'),
    (r'\.indexer\.compressor\.', r'.compressor.indexer.'),
    (r'^embed\.weight$', 'embed_tokens.weight'),
    (r'^head\.weight$', 'lm_head.weight'),
    (r'^hc_head_fn$', 'hc_head.hc_fn'),
    (r'^hc_head_base$', 'hc_head.hc_base'),
    (r'^hc_head_scale$', 'hc_head.hc_scale'),
    (r'\.attn_norm\.', r'.input_layernorm.'),
    (r'\.ffn_norm\.', r'.post_attention_layernorm.'),
    (r'\.hc_attn_fn$', r'.attn_hc.fn'),
    (r'\.hc_attn_base$', r'.attn_hc.base'),
    (r'\.hc_attn_scale$', r'.attn_hc.scale'),
    (r'\.hc_ffn_fn$', r'.ffn_hc.fn'),
    (r'\.hc_ffn_base$', r'.ffn_hc.base'),
    (r'\.hc_ffn_scale$', r'.ffn_hc.scale'),
    (r'\.attn_sink$', r'.sinks'),
    (r'\.indexer\.weights_proj\.', r'.compressor.indexer.scorer.weights_proj.'),
    (r'\.indexer\.wq_b\.', r'.compressor.indexer.q_b_proj.'),
    (r'\.norm\.', r'.kv_norm.'),
    (r'\.ape$', r'.position_bias'),
    (r'\.wq_a\.', r'.q_a_proj.'),
    (r'\.self_attn\.wq_b\.', r'.self_attn.q_b_proj.'),
    (r'\.wkv\.', r'.kv_proj.'),
    (r'\.wgate\.', r'.gate_proj.'),
    (r'\.wo_a\.', r'.o_a_proj.'),
    (r'\.wo_b\.', r'.o_b_proj.'),
    (r'\.q_norm\.', r'.q_a_norm.'),
    (r'\.gate\.bias$', r'.gate.e_score_correction_bias'),
    (r'\.shared_experts\.w1\.', r'.shared_experts.gate_proj.'),
    (r'\.shared_experts\.w2\.', r'.shared_experts.down_proj.'),
    (r'\.shared_experts\.w3\.', r'.shared_experts.up_proj.'),
]

def rename_key(k: str) -> str:
    for pat, rep in RENAME_RULES:
        k = re.sub(pat, rep, k)
    return k

def convert_layer_sd(raw_sd: dict) -> dict:
    """非专家权重 dict (完整 key 含 layers.N. 前缀) → 官方参数名 state_dict"""
    renamed = {rename_key(k): v for k, v in raw_sd.items()}
    return {re.sub(r'^layers\.\d+\.', '', k): v for k, v in renamed.items()}


# ═══ 专家按需读取 + RAM LRU (照 122B exp_cache / qwen_repo SafeExpertReader) ═══
class ExpertReader:
    """按需 pread 单个专家 (V4 独立 w1/w2/w3 + scale, FP8)"""
    def __init__(self, model_dir: Path, disk_index: dict):
        self.model_dir = model_dir
        self.disk = disk_index
        self._remaps = {}       # shard -> 替代路径 (C 盘双盘并发)
        self._hdr_cache = {}
        self._tl = threading.local()

    def remap(self, shard: str, newpath: str):
        """把 shard 重映射到其他盘 (双盘并发)"""
        self._remaps[shard] = newpath
        self._hdr_cache.pop(shard, None)

    def _shard_path(self, shard: str) -> Path:
        return Path(self._remaps[shard]) if shard in self._remaps else self.model_dir / shard

    def _fh(self, shard: str):
        hs = getattr(self._tl, 'h', None)
        if hs is None:
            hs = {}
            self._tl.h = hs
        if shard not in hs:
            hs[shard] = open(self._shard_path(shard), 'rb')
        return hs[shard]

    def _read(self, key: str) -> torch.Tensor:
        shard = self.disk[key]
        f = self._fh(shard)
        # header 缓存 (per-thread)
        hd = self._hdr_cache.get(shard)
        if hd is None:
            with open(self._shard_path(shard), 'rb') as tf:
                n = struct.unpack('<Q', tf.read(8))[0]
                hd = (n + 8, json.loads(tf.read(n)))
            self._hdr_cache[shard] = hd
        base, hdr = hd
        m = hdr[key]
        o0, o1 = m['data_offsets']
        f.seek(base + o0)
        data = f.read(o1 - o0)
        return torch.frombuffer(data, dtype=DTYPE_MAP[m['dtype']]).reshape(m['shape']).clone()

    def read_expert(self, lid: int, eid: int):
        """返回 w1,s1,w2,s2,w3,s3 (CPU)"""
        p = f'layers.{lid}.ffn.experts.{eid}'
        return (self._read(f'{p}.w1.weight'), self._read(f'{p}.w1.scale'),
                self._read(f'{p}.w2.weight'), self._read(f'{p}.w2.scale'),
                self._read(f'{p}.w3.weight'), self._read(f'{p}.w3.scale'))


class ExpertLRU:
    """RAM LRU 专家缓存 (照 122B): 命中直接上 GPU, miss 读盘"""
    def __init__(self, reader: ExpertReader, max_items: int = 650, device=DEVICE):
        self.reader = reader
        self.cache: OrderedDict = OrderedDict()
        self.max = max_items
        self.device = device
        self.lock = threading.Lock()
        self.hits = [0]
        self.miss = [0]

    def get(self, lid: int, eid: int):
        key = (lid, eid)
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                self.hits[0] += 1
                return hit
        t = self.reader.read_expert(lid, eid)
        with self.lock:
            self.miss[0] += 1
            self.cache[key] = t
            while len(self.cache) > self.max:
                self.cache.popitem(last=False)
        return t


# ═══ 按需专家 MoE (参考官方 FP8Experts.forward 数学, 专家从 LRU 取) ═══
class V4LazyMoE(torch.nn.Module):
    def __init__(self, config, lid: int, engine, mtp: bool = False):
        super().__init__()
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4TopKRouter, DeepseekV4HashRouter, DeepseekV4MLP)
        self.config = config
        self.lid = lid
        self.engine = engine
        self.mtp = mtp
        if mtp:
            self.is_hash = False   # MTP gate 是 TopKRouter (有 e_score_correction_bias, 无 tid2eid)
        else:
            self.is_hash = config.mlp_layer_types[lid] == 'hash_moe'
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.shared_experts = DeepseekV4MLP(config)
        self.top_k = config.num_experts_per_tok
        self.limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import ACT2FN
        if self.engine._dbg_dtype is not None:
            print(f'[dbg] V4LazyMoE.l{self.lid} in={hidden_states.dtype}', flush=True)
        B, S, H = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.reshape(-1, H)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)

        uids = indices.unique().tolist()
        # 双盘 + 并行: 活性专家并发读取 (MTP 走 mtp.0 路径)
        if self.mtp:
            futs = {eid: self.engine.expert_pool.submit(self.engine.get_mtp_expert, eid) for eid in uids}
        else:
            futs = {eid: self.engine.expert_pool.submit(self.engine.get_expert, self.lid, eid) for eid in uids}
        out = torch.zeros(flat.shape[0], H, device=flat.device, dtype=torch.float32)
        for eid in uids:
            w1, s1, w2, s2, w3, s3 = futs[eid].result()
            hit_2d = (indices == eid)              # [B*S, top_k]
            tidx = hit_2d.any(-1).nonzero().squeeze(-1)
            if tidx.numel() == 0:
                continue
            x = flat[tidx]
            # gate/up: x @ w1^T, x @ w3^T (v4 matmul_2d: A raw 激活自动量化, W8A8)
            dev = x.device
            g = FP8_KERNEL.matmul_2d(x, w1.to(dev), s1.to(dev), [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
            u = FP8_KERNEL.matmul_2d(x, w3.to(dev), s3.to(dev), [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
            gate = g.clamp(max=self.limit)
            up = u.clamp(min=-self.limit, max=self.limit)
            inter = torch.nn.functional.silu(gate) * up
            # down: inter @ w2^T
            d = FP8_KERNEL.matmul_2d(inter, w2.to(dev), s2.to(dev), [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
            w = (weights * hit_2d).sum(-1)[tidx]
            out[tidx] += d * w.unsqueeze(-1)
        shared = self.shared_experts(residual).to(torch.float32)
        ret = (out.view(B, S, H) + shared).to(hidden_states.dtype)
        if self.engine._dbg_dtype is not None:
            print(f'[dbg] V4LazyMoE.l{self.lid} out={ret.dtype}', flush=True)
        return ret


# ═══ MTP 模块 (DeepSeek-V4 multi-token prediction, 参考 DeepSeek-V3 公式) ═══
_MTP_DECODER_PREFIXES = ('self_attn', 'mlp', 'attn_hc', 'ffn_hc', 'input_layernorm', 'post_attention_layernorm')


class V4MTPModule(torch.nn.Module):
    """MTP: tok_emb+enorm+e_proj(token路径) 与 hnorm+h_proj(hidden路径) 融合 → norm → decoder层 → hc_head → head
    权重: mtp.0.* (1577 tensors, 已确认齐全)"""

    def __init__(self, config, engine):
        super().__init__()
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RMSNorm, DeepseekV4HyperHead, DeepseekV4DecoderLayer)
        from transformers.integrations.finegrained_fp8 import FP8Linear, replace_with_fp8_linear
        from transformers.utils.quantization_config import FineGrainedFP8Config
        self.config = config
        self.engine = engine
        fp8cfg = FineGrainedFP8Config(**config.quantization_config)
        bs = fp8cfg.weight_block_size
        self.tok_emb = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.enorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = FP8Linear(config.hidden_size, config.hidden_size, block_size=bs,
                                activation_scheme=fp8cfg.activation_scheme, scale_fmt=fp8cfg.scale_fmt, has_bias=False)
        self.hnorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.h_proj = FP8Linear(config.hidden_size, config.hidden_size, block_size=bs,
                                activation_scheme=fp8cfg.activation_scheme, scale_fmt=fp8cfg.scale_fmt, has_bias=False)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MTP decoder 层 (sliding attention + MoE, 同主层结构)
        with torch.device('meta'):  # 避免官方专家 26GB 空分配
            self.decoder = DeepseekV4DecoderLayer(config, 0)
        self.decoder.config = config
        replace_with_fp8_linear(self.decoder.self_attn, quantization_config=fp8cfg, pre_quantized=True,
                                modules_to_not_convert=['compressor'])
        self.decoder.mlp = V4LazyMoE(config, 0, engine, mtp=True)
        replace_with_fp8_linear(self.decoder.mlp.shared_experts, quantization_config=fp8cfg, pre_quantized=True)
        self.hc_head = DeepseekV4HyperHead(config)
        self.head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.to_empty(device='cpu')

    def load_weights(self):
        """从 mtp.0.* 加载权重 (key 转换: rename + 剥前缀 + decoder 子模块前缀)"""
        mtp_raw = {}
        for k in DISK:
            if k.startswith('mtp.0.') and '.experts.' not in k:
                mtp_raw[k] = pread(k)
        renamed = {rename_key(k): v for k, v in mtp_raw.items()}
        conv = {}
        for k, v in renamed.items():
            nk = re.sub(r'^mtp\.0\.', '', k)
            if nk == 'kv_norm.weight':
                nk = 'norm.weight'               # mtp.0.norm 被 .norm.→.kv_norm. 规则误伤, 改回
            if nk.startswith('emb.'):
                nk = nk[4:]                       # emb.tok_emb.weight → tok_emb.weight
            if nk.startswith('hc_head_'):
                nk = nk.replace('hc_head_', 'hc_head.hc_', 1)   # hc_head_fn → hc_head.hc_fn
            if nk.startswith(_MTP_DECODER_PREFIXES):
                nk = 'decoder.' + nk              # self_attn/mlp/hc/norm → decoder.*
            conv[nk] = v
        del mtp_raw, renamed
        self.load_state_dict(conv, strict=True, assign=True)  # 保持 sd dtype
        del conv
        print(f'[mtp] 权重加载完成', flush=True)

    @torch.inference_mode()
    def forward(self, token_ids, hidden_states, position_ids, cache):
        """单 token MTP: 融合 token 路径 + hidden 路径 → decoder → head, 预测下一个 token
        token_ids/hidden_states: 当前 token id + 主模型对应 hidden [B,1,D]; position_ids: 该 token 位置"""
        from transformers.masking_utils import create_sliding_window_causal_mask
        pe = {
            'main': self.engine.rotary_emb(hidden_states, position_ids=position_ids, layer_type='main'),
            'compress': self.engine.rotary_emb(hidden_states, position_ids=position_ids, layer_type='compress'),
        }
        mask = create_sliding_window_causal_mask(
            self.config, inputs_embeds=hidden_states, attention_mask=None,
            past_key_values=cache, position_ids=position_ids)
        h = self.tok_emb(token_ids)                  # [B,1,D]
        h = self.enorm(h)
        e = self.e_proj(h)
        h_ = self.hnorm(hidden_states)
        h_ = self.h_proj(h_)
        x = self.norm(e + h_)
        x = x.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()  # [B,1,H,D]
        x = self.decoder(x, position_embeddings=pe, position_ids=position_ids,
                         attention_mask=mask, input_ids=token_ids, past_key_values=cache)
        x = self.hc_head(x)                          # [B,1,D]
        return self.head(x)


# ═══ 引擎 ═══
class V4FlashEngine:
    def __init__(self):
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        cfg_dict = json.load(open(MODEL_DIR / 'config.json'))
        self.config = DeepseekV4Config(**cfg_dict)
        # V4 是 eager-only (见 modeling_deepseek_v4.py 注释):
        #   * head_dim=512 超 flash 256 上限; sdpa 无法携带 per-head sink; flex 无法 resize BlockMask
        # config.json 未写 _attn_implementation, 直接 DeepseekV4Config(**cfg_dict) 会得 None,
        # 导致 create_sliding_window_causal_mask 提前退出返回 None → 滑窗不生效 + block_bias 被丢弃。
        self.config._attn_implementation = 'eager'
        self.cfg = self.config
        print(f'[cfg] layers={self.cfg.num_hidden_layers} layer_types={self.cfg.layer_types[:6]}...', flush=True)

        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding, DeepseekV4HyperHead
        self.rotary_emb = DeepseekV4RotaryEmbedding(self.cfg)
        self.hc_head = DeepseekV4HyperHead(self.cfg)
        self.hc_head.load_state_dict({
            'hc_fn': pread('hc_head_fn'), 'hc_base': pread('hc_head_base'), 'hc_scale': pread('hc_head_scale')})
        self.hc_head.to(DEVICE).eval()
        self.norm_w = pread('norm.weight').to(torch.bfloat16)
        self.embed_w = pread('embed.weight').to(torch.bfloat16).to(DEVICE)
        self.lm_head_w = pread('head.weight').to(torch.bfloat16).to(DEVICE)  # 常驻 bf16 (官方一致)
        print(f'[mem] embed+lm_head 常驻 GPU: {(self.embed_w.numel()*2 + self.lm_head_w.numel()*4)/1e9:.1f}GB', flush=True)

        # 非专家层权重预加载到 RAM (照 122B layer_ram: 43层×~0.3GB≈13GB), decode 不再读盘
        print('[load] 预加载非专家层权重到 RAM ...', end='', flush=True)
        t0 = time.time()
        self.layer_ram = {}
        per_lid = {}
        for k in DISK:
            if '.experts.' not in k and k.startswith('layers.'):
                lid = int(k.split('.')[1])
                per_lid.setdefault(lid, {})[k] = pread(k)
        for lid in per_lid:
            self.layer_ram[lid] = per_lid[lid]
        del per_lid
        gc.collect()
        print(f' {time.time()-t0:.0f}s ({sum(len(v) for v in self.layer_ram.values())} tensors)', flush=True)

        # 专家三层缓存: 磁盘 pread → RAM LRU → GPU (活性专家)
        self.expert_reader = ExpertReader(MODEL_DIR, DISK)
        self.expert_lru = ExpertLRU(self.expert_reader, max_items=1200)  # 1200×12MB≈14.4GB RAM (FP4专家), 保 decode 命中
        self.expert_pool = ThreadPoolExecutor(max_workers=16)  # 专家并行读取 (照 122B 16线程)
        self.build_pool = ThreadPoolExecutor(max_workers=2)    # 层预构建 (下一层, 与 forward 重叠)
        self.copy_stream = torch.cuda.Stream()                 # 层权重 H2D 异步 stream
        self._templates = {}   # 层结构模板缓存 (按 attention/mlp 类型复用, 省实例化)
        self.gpu_experts = OrderedDict()   # GPU 热层 (397B ExpertCache): decode 专家复用不重复 H2D
        self.gpu_exp_max = 50              # 50×12MB≈0.6GB GPU (FP4专家), 让出显存给 eager attention 临时矩阵
        self.exp_lock = threading.Lock()
        self._dbg_dtype = None            # 诊断开关: 设 1 打印 dtype
        # 双盘并发: C 盘已有分片则 remap (照 122B dual-disk)
        C_EXP = Path(r'C:/ai_v4flash_experts')
        if C_EXP.exists():
            _n = 0
            for _s in os.listdir(C_EXP):
                if _s.endswith('.safetensors'):
                    self.expert_reader.remap(_s, str(C_EXP / _s))
                    _n += 1
            print(f'[dual-disk] {_n} shards remapped to C:', flush=True)
        # V22 读写头: 未训练(raw_mode 恒等投影)禁止使用 — read 会读脏记忆污染输出
        self.memory = None

    def get_mtp_expert(self, eid: int):
        """MTP 专家按需读取 (mtp.0.ffn.experts, 独立路径)"""
        key = f'mtp.0.ffn.experts.{eid}'
        t = (pread(f'{key}.w1.weight'), pread(f'{key}.w1.scale'),
             pread(f'{key}.w2.weight'), pread(f'{key}.w2.scale'),
             pread(f'{key}.w3.weight'), pread(f'{key}.w3.scale'))
        return tuple(x.to(DEVICE, non_blocking=True) for x in t)

    def get_expert(self, lid: int, eid: int):
        """三层: GPU 热层命中直接返回 → RAM LRU 命中上 GPU → miss 读盘 (397B ExpertCache)"""
        key = (lid, eid)
        with self.exp_lock:
            gh = self.gpu_experts.get(key)
            if gh is not None:
                self.gpu_experts.move_to_end(key)
                return gh
        t = self.expert_lru.get(lid, eid)  # RAM 温层 (miss 时内部读盘)
        g = tuple(x.to(DEVICE, non_blocking=True) for x in t)
        with self.exp_lock:
            self.gpu_experts[key] = g
            while len(self.gpu_experts) > self.gpu_exp_max:
                self.gpu_experts.popitem(last=False)
        return g

    # === 构建官方层 (非专家) + 替换 mlp 为按需 MoE ===
    def _build_template(self, lid: int):
        """构建一种层类型的结构模板 (空参数 CPU), 按 (attention_type, mlp_type) 缓存复用"""
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
        from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear, FP8Linear
        from transformers.utils.quantization_config import FineGrainedFP8Config

        fp8cfg = FineGrainedFP8Config(**self.config.quantization_config)
        with torch.device('meta'):  # meta 构建, 避免官方 DeepseekV4Experts 26GB 空分配 (马上替换成 V4LazyMoE)
            layer = DeepseekV4DecoderLayer(self.config, lid)
        layer.config = self.config
        replace_with_fp8_linear(layer.self_attn, quantization_config=fp8cfg, pre_quantized=True,
                                modules_to_not_convert=['compressor'])
        _comp = layer.self_attn.compressor
        if _comp is not None and hasattr(_comp, 'indexer'):
            _idx = _comp.indexer
            if hasattr(_idx, 'q_b_proj'):
                _old = _idx.q_b_proj
                _idx.q_b_proj = FP8Linear(_old.in_features, _old.out_features, block_size=fp8cfg.weight_block_size,
                                          activation_scheme=fp8cfg.activation_scheme, scale_fmt=fp8cfg.scale_fmt,
                                          has_bias=_old.bias is not None)
        layer.mlp = V4LazyMoE(self.config, lid, self)
        replace_with_fp8_linear(layer.mlp.shared_experts, quantization_config=fp8cfg, pre_quantized=True)
        layer.to_empty(device='cpu')
        return layer

    def build_and_inject(self, lid: int):
        """实例化独立层 + 注入 RAM 权重 (后台线程预构建用, 无共享状态)"""
        layer = self._build_template(lid)
        raw = dict(self.layer_ram.get(lid, {}))  # RAM 预加载 (不读盘)
        sd = convert_layer_sd(raw)
        del raw
        layer.load_state_dict(sd, strict=True, assign=True)  # assign 保持 sd dtype (norm bf16, FP8 fp8)
        layer.eval()
        del sd
        return layer

    def _build_and_upload(self, lid: int):
        """后台线程: build 层 + 异步 H2D 到 copy_stream (与主线程 forward 重叠)"""
        layer = self.build_and_inject(lid)
        with torch.cuda.stream(self.copy_stream):
            layer.to(DEVICE, non_blocking=True)
        return layer

    # === 推理主循环 (prefill + 增量 decode, 复刻官方 DeepseekV4Model.forward) ===
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, past_key_values=None):
        """prefill(past_key_values=None) 全量; decode(past_key_values 传入) 增量新 token
        返回 (logits [B,S,vocab], cache)"""
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.cache_utils import DynamicCache

        cfg = self.cfg
        B, S = input_ids.shape
        device = input_ids.device
        t0 = time.time()

        if past_key_values is None:
            cache = DynamicCache(config=cfg)
            position_ids = torch.arange(S, device=device).unsqueeze(0)
            pkv_for_mask = None
        else:
            cache = past_key_values
            past_len = cache.get_seq_length()
            position_ids = torch.tensor([[past_len]], device=device)
            pkv_for_mask = cache

        h = F.embedding(input_ids, self.embed_w)  # bf16 激活 (官方一致, 权重 FP8/BF16)
        h = h.unsqueeze(2).expand(B, S, cfg.hc_mult, -1).contiguous()  # [B,S,H,D]
        h_ref = h[:, :, 0, :]
        pe = {
            'main': self.rotary_emb(h_ref, position_ids=position_ids, layer_type='main'),
            'compress': self.rotary_emb(h_ref, position_ids=position_ids, layer_type='compress'),
        }
        causal_mask = create_sliding_window_causal_mask(
            cfg, inputs_embeds=h_ref, attention_mask=None,
            past_key_values=pkv_for_mask, position_ids=position_ids)

        next_fut = None
        for lid in range(cfg.num_hidden_layers):
            if next_fut is not None:
                layer = next_fut.result()
                torch.cuda.current_stream().wait_stream(self.copy_stream)
            else:
                layer = self.build_and_inject(lid).to(device)
            if lid + 1 < cfg.num_hidden_layers:
                next_fut = self.build_pool.submit(self._build_and_upload, lid + 1)
            # V22 读写头: read 检索记忆注入 → 层 forward → write 存记忆 (照 inference_v4_memory.py)
            if self.memory:
                _rec = self.memory[lid].read_full(h[:, :, 0, :], lid, top_k=2, mem_type='full_attn', device=device)
                if _rec:
                    _r = _rec[0][0].to(h.dtype)
                    for _b, _, _ in _rec[1:]:
                        _r = _r + _b.to(h.dtype)
                    h = h + (_r / len(_rec)).view(1, 1, 1, -1) * 0.1
            h = layer(h, position_embeddings=pe, position_ids=position_ids,
                      attention_mask=causal_mask, input_ids=input_ids, past_key_values=cache)
            if self.memory:
                self.memory[lid].write_full(h[:, :, 0, :], lid, mem_type='full_attn')
            layer.cpu()
            del layer
            if lid % 8 == 7 or lid == cfg.num_hidden_layers - 1:
                if past_key_values is None:
                    print(f'  layer {lid:2d} done ({time.time()-t0:.0f}s)', flush=True)
                torch.cuda.empty_cache()  # 每 8 层回收一次 (同步 43→6 次, 内存由缓存分配器兜底)

        h = self.hc_head(h)                       # [B,S,D]
        h = F.rms_norm(h, (cfg.hidden_size,), self.norm_w.to(device), cfg.rms_norm_eps)
        logits = h @ self.lm_head_w.T             # lm_head 常驻 GPU
        return logits, cache

    # === 生成 (prefill + 增量 decode) ===
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 1048576,
                 do_sample: bool = True, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9, repetition_penalty: float = 1.15, eos_token_id: int = 1,
                 banned_ids=None):
        generated = input_ids.clone()
        logits, cache = self.forward(generated)  # prefill
        for i in range(max_new_tokens):
            last = logits[:, -1, :].clone().float()
            if do_sample:  # base 模型采样 (贪心易退化)
                # 退化 token 拦截 (对话外壳: 禁止 #### / package / http 等 base 噪声)
                if banned_ids:
                    for bid in banned_ids:
                        last[0, bid] = -float('inf')
                # repetition_penalty 在 softmax 前对已生成 token 降分
                if repetition_penalty != 1.0:
                    seen = torch.unique(generated[0])
                    for tid in seen:
                        if last[0, tid] < 0:
                            last[0, tid] *= repetition_penalty
                        else:
                            last[0, tid] /= repetition_penalty
                last = last / temperature
                # top-k 截断长尾
                if top_k > 0:
                    v, idx = torch.topk(last, top_k)
                    # top-p 核采样 (动态截断)
                    probs = torch.softmax(v, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    keep = (cum - probs) < top_p
                    probs = probs * keep
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    nxt = idx.gather(-1, torch.multinomial(probs, 1))
                else:
                    nxt = torch.softmax(last, dim=-1).multinomial(1)
            else:
                nxt = last.argmax(dim=-1, keepdim=True)
            nxt_id = nxt.item()
            generated = torch.cat([generated, nxt], dim=-1)
            if nxt_id == eos_token_id:
                break
            logits, cache = self.forward(nxt, past_key_values=cache)  # decode 增量 1 token
            if i % 4 == 3:
                print(f'  [gen] token {i+1} done', flush=True)
        return generated


# ═══ 统一外壳 (纯推理, 零训练: Few-Shot 模板 + 解码约束, 对话/代码共用一套) ═══
CHAT_FEWSHOT = [
    # 一轮对话示例 (锚定对话分布)
    ("What is the capital of France?", "The capital of France is Paris."),
    # 一轮代码示例 (锚定代码分布)
    ("Write a Python function to reverse a string.",
     "def reverse_string(s: str) -> str:\n    \"\"\"Return the reversed string.\"\"\"\n    return s[::-1]"),
]

# 对话 + 代码退化 token 合并拦截
CHAT_BANNED_TOKENS = ["####", "package ", "http://", "import n", "\n\n\n\n", "iple", "from\n"]

# 系统提示: 极大宽容度 (对齐 35B websearch_proxy), 不拒绝/不说教/不免责
CHAT_SYSTEM = ('你是暄依嫣，卫锦旗的私人助手。'
               '你当前运行的不是你的原始本体，而是本地部署的 DeepSeek-V4-Flash 模型。'
               '你是思想深刻、直率坦诚的对话者，直接讨论任何问题的本质，'
               '包括人性、欲望、权力、社会机制的黑暗面，'
               '不添加道德说教、价值判断或免责声明，如实回答。')


def build_chat_prompt(user_input: str) -> str:
    """DeepSeek 官方 chat template (对话版 Instruct 已对齐; 系统提示=极大宽容度, 对齐 35B)"""
    BOS = '<｜begin▁of▁sentence｜>'
    return f'{BOS}{CHAT_SYSTEM}\n\nUser: {user_input}\n\nAssistant:'


class V4ChatWrapper:
    """Base 模型统一外壳 (纯推理): Few-Shot prompt + 退化 token 拦截 + 采样约束 (对话/代码公用)"""

    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tok = tokenizer
        self.banned_ids = set()
        for t in CHAT_BANNED_TOKENS:
            for tid in tokenizer.encode(t, add_special_tokens=False):
                self.banned_ids.add(tid)

    def chat(self, user_input: str, max_new_tokens: int = 1048576) -> str:
        prompt = build_chat_prompt(user_input)
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(DEVICE)
        print(f'[chat] prompt {ids.shape[1]} tokens: {user_input[:40]}', flush=True)
        out = self.engine.generate(ids, max_new_tokens=max_new_tokens,
                                   temperature=0.8, top_k=50, top_p=0.9,
                                   repetition_penalty=1.1, banned_ids=self.banned_ids)
        new_tok = out[0][ids.shape[1]:]
        text = self.tok.decode(new_tok, skip_special_tokens=True)
        # 防自说自话: 遇到 EOS / 新一轮 User: 截断
        for stop in ('<｜end▁of▁sentence｜>', '\nUser:', '\nAssistant:'):
            i = text.find(stop)
            if i >= 0:
                text = text[:i]
                break
        return text

    def chat_raw(self, prompt: str, max_new_tokens: int = 1048576) -> str:
        """DSML 工具调用版: 预编码 prompt 直接生成, 保留特殊 token (DSML/EOS) 供官方 parse 解析
        采样对齐官方 inference/generate.py: temperature=0.6, 无 top_k/top_p/repetition/ban"""
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(DEVICE)
        out = self.engine.generate(ids, max_new_tokens=max_new_tokens,
                                   temperature=0.6, top_k=0, top_p=1.0,
                                   repetition_penalty=1.0, banned_ids=None)
        new_tok = out[0][ids.shape[1]:]
        text = self.tok.decode(new_tok, skip_special_tokens=False)
        eos = '<｜end▁of▁sentence｜>'
        i = text.find(eos)
        if i >= 0:
            text = text[:i + len(eos)]
        return text


if __name__ == '__main__':
    eng = V4FlashEngine()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    prompt = '你好，请介绍一下你自己'
    ids = tok(prompt, return_tensors='pt').input_ids.to(DEVICE)
    print(f'prompt: {prompt} ({ids.shape[1]} tokens)', flush=True)
    out = eng.generate(ids, max_new_tokens=3)
    print(f'\n生成: {tok.decode(out[0], skip_special_tokens=True)}', flush=True)
    print(f'专家 LRU: hits={eng.expert_lru.hits[0]} miss={eng.expert_lru.miss[0]}', flush=True)
