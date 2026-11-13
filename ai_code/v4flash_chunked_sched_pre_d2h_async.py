"""V4-Flash 引擎 v3 — 三层缓存 (照 122B test_122b_fast.py / qwen_repo 基建)
- 非专家权重: 逐层 pread → RAM → 官方层 forward (单层放 GPU, 算完释放)
- 专家: PreadExpertReader 按需 pread 活性 top-6 + RAM LRU (OrderedDict), 不打包 3D
- forward 数学: attention/mHC/gate/shared 用官方 DeepseekV4DecoderLayer; 专家 FP8 GEMM 手写
  (参考官方 FP8Experts.forward 数学), 全程 fp32 激活
峰值: embed 1GB + 单层非专家 ~0.3GB + 活性专家 ~0.2GB ≈ 1.5GB GPU / <2GB CPU

【2026-08-17 魔改】添加分块注意力支持，根治长上下文 OOM
"""
import os, sys, json, struct, time, re, gc, threading, copy, ctypes
from concurrent.futures import ThreadPoolExecutor

# === 分块注意力补丁（在线分块 softmax，支持 1M 上下文）===
from v4flash_chunked_attention import apply_chunked_attention_patch, remove_chunked_attention_patch, CPUDynamicCache
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

    def layer_shard(self, lid: int) -> str:
        """层 lid 的专家所在 shard 名"""
        return self.disk[f'layers.{lid}.ffn.experts.0.w1.weight']

    def on_c(self, shard: str) -> bool:
        """该 shard 是否已 remap 到 C 盘"""
        return shard in self._remaps

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
        # 并行: 活性专家并发读取 (MTP 走 mtp.0 路径)
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
            # gate/up: x @ w1^T, x @ w3^T (w1/w2/w3 已由 get_expert 保证在 GPU, 无需 .to(dev))
            g = FP8_KERNEL.matmul_2d(x, w1, s1, [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
            u = FP8_KERNEL.matmul_2d(x, w3, s3, [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
            gate = g.clamp(max=self.limit)
            up = u.clamp(min=-self.limit, max=self.limit)
            inter = torch.nn.functional.silu(gate) * up
            # down: inter @ w2^T
            d = FP8_KERNEL.matmul_2d(inter, w2, s2, [BLOCK_SIZE, BLOCK_SIZE], torch.bfloat16).float()
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
        # mtp.0 权重里没有 tok_emb 和 head — DeepSeek-V4 的 MTP 复用主模型的 embedding 和 lm_head
        conv['tok_emb.weight'] = self.engine.embed_w
        conv['head.weight'] = self.engine.lm_head_w
        self.load_state_dict(conv, strict=True, assign=True)  # 保持 sd dtype
        del conv
        # 移到 GPU (tok_emb/head 共享主模型已在 GPU; decoder 层从 CPU 移到 GPU, 避免 cuda/cpu 不匹配)
        self.to(DEVICE)
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
        return self.head(x), x                       # (logits, MTP hidden) — hidden 供下一级 MTP 链式复用


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

        # === 分块注意力配置（面向 1M 上下文极限）===
        # 12GB 显存紧张 → 尽早分块：阈值 1024（超过即分块），分块本身显存恒定
        self.chunked_attention = True   # 是否启用分块注意力
        self.max_sequence_length = 1024  # 超过此长度触发在线分块（12G 显存尽早分块）
        self.chunk_size = 256           # query 分块（token）
        self.kv_chunk_size = 512        # KV 分块（token）— 长 KV 时流式分块处理
        self.prefill_chunk_size = 1024  # 流式 prefill 段大小（token）：超长输入分段，hidden_states 不撑爆 GPU
        # 说明: KV 保持 GPU DynamicCache（单头 MQA，1M 上下文 ≈ 1GB 可容纳），
        #       在线分块不展开 KV（MQA 广播），避免 [B,64,S,S] 矩阵爆炸
        #       CPUDynamicCache 类已就绪，后续如需 KV 下放 DRAM 可接入

        if self.chunked_attention:
            success = apply_chunked_attention_patch(
                max_sequence_length=self.max_sequence_length,
                chunk_size=self.chunk_size,
                kv_chunk_size=self.kv_chunk_size,
                use_online=True
            )
            if not success:
                print("[WARN] 分块注意力补丁失败，长序列可能 OOM", flush=True)
        else:
            print("[chunked_attention] 分块注意力已禁用", flush=True)
        print(f"[chunked_attention] Q分块={self.chunk_size}, KV分块={self.kv_chunk_size}, "
              f"触发阈值={self.max_sequence_length} tokens", flush=True)

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

        # 非专家层权重: 默认不预加载到 RAM, 靠页缓存(数据流通道)缓存模型文件
        # ⚠️ 2026-08-25 优化: 预加载13GB进RAM会挤掉页缓存 → 磁盘流式读取变慢(内存墙)
        #    改为 build 时按需 pread → 走页缓存, 进程少占13GB, 给数据流通道腾位置
        #    可逆: use_ram_preload=True 恢复旧行为(预加载进RAM, 不读盘)
        self.use_ram_preload = True
        self.layer_ram = {}   # 空: build 时按需 pread; 若 use_ram_preload=True 则预加载填充
        # 构建层→非专家key索引 (只存key名, 不加载权重, build时按索引pread)
        from collections import defaultdict
        self.layer_keys = defaultdict(list)
        for k in DISK:
            if '.experts.' not in k and k.startswith('layers.'):
                self.layer_keys[int(k.split('.')[1])].append(k)
        self.layer_keys = dict(self.layer_keys)
        if self.use_ram_preload:
            print('[load] 预加载非专家层权重到 RAM ...', end='', flush=True)
            t0 = time.time()
            self.layer_sd = {}   # 转换后的官方 state_dict (避免每层重复 convert_layer_sd)
            per_lid = {}
            for lid, keys in self.layer_keys.items():
                per_lid[lid] = {k: pread(k) for k in keys}
            for lid in per_lid:
                self.layer_ram[lid] = per_lid[lid]
                self.layer_sd[lid] = convert_layer_sd(per_lid[lid])  # 预转换缓存
            del per_lid
            gc.collect()
            print(f' {time.time()-t0:.0f}s ({sum(len(v) for v in self.layer_ram.values())} tensors)', flush=True)
        else:
            self.layer_sd = {}
            print(f'[load] 非专家权重走页缓存(不预加载RAM) {sum(len(v) for v in self.layer_keys.values())} tensors', flush=True)

        # 一次性构建 43 层层缓存 (forward 复用, 消除每轮 rebuild 5s)
        # 层对象常驻 RAM, forward 只做 H2D 上传; 权重已注入层对象, 释放原始 dict 省内存
        print('[load] 构建 43 层缓存 ...', end='', flush=True)
        t0 = time.time()
        self.layers = [self.build_and_inject(lid) for lid in range(self.cfg.num_hidden_layers)]
        if hasattr(self, 'layer_ram'):
            del self.layer_ram
        if hasattr(self, 'layer_sd'):
            del self.layer_sd
        gc.collect()
        print(f' {time.time()-t0:.0f}s', flush=True)

        # 专家三层缓存: 磁盘 pread → RAM LRU → GPU (活性专家)
        self.expert_reader = ExpertReader(MODEL_DIR, DISK)
        self.expert_lru = ExpertLRU(self.expert_reader, max_items=1200)  # 1200×12MB≈14.4GB RAM 温层 (1500实测无提升,专家分布散)
        # 线程池: i5-14600KF 20线程, MTP 场景线程过多会饥饿(E-core 竞争) → expert 12 / build 1
        # 单池12线程 (双盘实验证明 32GB 内存扛不住两盘页缓存→交换; 8+8/4+4 双池均失败)
        self.expert_pool = ThreadPoolExecutor(max_workers=4)   # 专家读取 (扫描最优: 命中率高4线程够)
        self.build_pool = ThreadPoolExecutor(max_workers=10)   # H2D预取 (扫描最优: 深预取10层掩盖上传)
        self.copy_stream = torch.cuda.Stream()                 # 层权重 H2D 异步 stream
        self._templates = {}   # 层结构模板缓存 (按 attention/mlp 类型复用, 省实例化)
        self.gpu_experts = OrderedDict()   # GPU 热层 (397B ExpertCache): decode 专家复用不重复 H2D
        # 在线分块注意力已把注意力显存压到 O(块)，GPU 大量空闲 → 专家可多驻留提升 decode 命中率
        # 2026-08-22 实测: 启动后仅用 4.2GB, 空闲 7.6GB → 加热到 400 (400×12MB=4.8GB), 留 ~2.8GB 给 KV+层临时
        # (此前 MTP 加入时误以为显存紧张降到 50, 实测 MTP decoder 很小, 无需降)
        self.gpu_exp_max = 400  # 实测甜点位: 400(+27%prefill/+12%decode); 600 显存过满反慢(VRAM 9.7GB); 50 是原值
        self.exp_lock = threading.Lock()
        self._dbg_dtype = None            # 诊断开关: 设 1 打印 dtype
        self._probe_sync = False          # sync 拆解探针: 窗口循环分段计时 (wait_fut/wait_stream/pin/fwd/d2h)
        # 2026-08-26 sync 探针定论: pin 是纯负优化!
        #   * pinned:  wait_fut=0.10 wait_stream=0.000 pin=0.74 fwd=3.3 d2h=1.7 → decode 6.0s/tok
        #   * no-pin:  wait_fut=0.16 wait_stream=0.000 pin=0.00 fwd=3.5 d2h=1.8 → decode 5.5s/tok (-8%)
        #   * wait_stream 恒 0 → 双缓冲已隐藏 H2D, pin 不加速任何东西, 白花 0.7s 主线程 pin_memory
        # 默认关闭 (保留 _pin_window 定义 + 开关, 供回归测试)
        self._pin_enabled = False
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

        # ═══ MTP 多 token 预测 (加速 decode: 一次预测 2 token) ═══
        # 2026-08-23 升级为"拒绝采样"模式: MTP 预测 B 的 softmax 置信度 >= 阈值才接受
        #   → 高置信(B被模型认可)接受, 低置信(可能猜错)拒绝, 拒绝时不进主模型 cache → 零污染
        # 相比完全禁用(纯6.96s/tok): 保留高置信的MTP加速, 规避长文本漂移风险
        # 阈值可调: mtp_min_conf (0=全接受≈原MTP, 1=全拒绝≈纯单token)
        # 独立 cache 存 MTP decoder 的 KV, 不与主模型 43 层冲突
        self.use_mtp = True
        self.mtp_min_conf = 0.5   # 拒绝采样置信度阈值: B的softmax概率≥此值才接受
        self.mtp_steps = 1   # MTP 级联预测额外 token 数 (复用 mtp.0, 1=预测1个/轮出2token; 2=预测2个/轮出3token但OOD质量崩坏勿用)
        if self.use_mtp:
            try:
                from transformers.cache_utils import DynamicCache
                self.mtp = V4MTPModule(self.config, self)
                self.mtp.load_weights()
                self.mtp_cache = DynamicCache(config=self.config)
                print(f'[mtp] MTP 模块就绪 (每轮 {self.mtp_steps}+1 个 token, 速度≈×{self.mtp_steps+1})', flush=True)
            except Exception as e:
                self.mtp = None
                print(f'[mtp] MTP 初始化失败(自动禁用): {str(e)[:120]}', flush=True)
        else:
            self.mtp = None

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
        """实例化独立层 + 注入权重 (后台线程预构建用, 无共享状态)
        权重来源: use_ram_preload=True → layer_ram(RAM预加载); False → 按需pread(走页缓存)"""
        layer = self._build_template(lid)
        if self.layer_sd:   # 预加载模式: 用预转换好的 sd (省掉 convert_layer_sd)
            sd = self.layer_sd.get(lid, {})
        else:               # 页缓存模式: 按索引 pread + 现场转换
            raw = {k: pread(k) for k in self.layer_keys.get(lid, [])}
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

    def _upload_layer(self, lid: int):
        """后台线程: 复用缓存的层, 只做 H2D 上传 (不重建, 消除每轮 build 开销)"""
        with torch.cuda.stream(self.copy_stream):
            self.layers[lid].to(DEVICE, non_blocking=True)
        return self.layers[lid]

    def _pin_window(self, _g: int, _ge: int):
        """窗口层参数 pin 到锁页内存 (让 .to(device, non_blocking=True) 真异步 DMA)
        窗口级 2.4GB pinned, 每窗口复用, .cpu() 后释放
        2026-08-26: 默认关闭 (_pin_enabled=False) — 探针证 pin 不加速 H2D(wait_stream 恒0), 纯浪费 0.7s"""
        if not getattr(self, '_pin_enabled', False):
            return
        for _lid in range(_g, _ge):
            for _p in self.layers[_lid].parameters():
                try:
                    if not _p.is_pinned():
                        _p.data = _p.data.pin_memory()
                except Exception:
                    pass

    # === 推理主循环 (prefill + 增量 decode, 复刻官方 DeepseekV4Model.forward) ===
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, past_key_values=None, return_hidden=False):
        """prefill(past_key_values=None) 全量; decode(past_key_values 传入) 增量新 token
        return_hidden=True 时额外返回最后一层 hidden (MTP 预测用) → (logits, cache, h)"""
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
            if S > 1:
                # 多 token 续段（流式 prefill：第二段起）— position_ids 连续递增
                position_ids = torch.arange(past_len, past_len + S, device=device).unsqueeze(0)
            else:
                # 单 token decode（增量 1 token）
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

        # 层缓存 + 驻留窗口 + 双缓冲流水线: 当前窗口forward时后台预上传下一窗口(H2D与GPU计算重叠)
        K = 8   # 驻留层数 (8层×0.3GB=2.4GB VRAM)
        _nw = (cfg.num_hidden_layers + K - 1) // K   # 窗口数
        # 预提交第0窗口上传 (先 pin 第0窗口层, 真异步 H2D)
        self._pin_window(0, min(K, cfg.num_hidden_layers))
        _next_futs = [self.build_pool.submit(self._upload_layer, _lid) for _lid in range(0, min(K, cfg.num_hidden_layers))]
        for _wi in range(_nw):
            _g = _wi * K
            _ge = min(_g + K, cfg.num_hidden_layers)
            if self._probe_sync:
                _pt = time.time()
            # 等当前窗口上传完成 (GPU 等 H2D, 但下一窗口已在后台上传)
            for _f in _next_futs:
                _f.result()
            if self._probe_sync:
                _t1 = time.time()
            torch.cuda.current_stream().wait_stream(self.copy_stream)
            if self._probe_sync:
                _t2 = time.time()
            # 提前 pin + 提交下一窗口上传 (与当前窗口 forward 重叠, 真异步)
            if _ge < cfg.num_hidden_layers:
                _nxt = _ge + K
                self._pin_window(_ge, min(_nxt, cfg.num_hidden_layers))
                _next_futs = [self.build_pool.submit(self._upload_layer, _lid)
                              for _lid in range(_ge, min(_nxt, cfg.num_hidden_layers))]
            if self._probe_sync:
                _t3 = time.time()
            # 连续 forward 本窗口 K 层
            for lid in range(_g, _ge):
                layer = self.layers[lid]
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
            if self._probe_sync:
                _t4 = time.time()
            # 整组回 CPU (D2H 一次/组)
            for lid in range(_g, _ge):
                self.layers[lid].cpu()
            if self._probe_sync:
                _t5 = time.time()
                _acc = self._sync_acc = getattr(self, '_sync_acc', {})
                _acc['wait_fut'] = _acc.get('wait_fut', 0.0) + (_t1 - _pt)
                _acc['wait_stream'] = _acc.get('wait_stream', 0.0) + (_t2 - _t1)
                _acc['pin'] = _acc.get('pin', 0.0) + (_t3 - _t2)
                _acc['fwd'] = _acc.get('fwd', 0.0) + (_t4 - _t3)
                _acc['d2h'] = _acc.get('d2h', 0.0) + (_t5 - _t4)
                _acc['win'] = _acc.get('win', 0) + 1
                print(f'  [sync] W{_wi}: wait_fut={_t1-_pt:.3f}s wait_stream={_t2-_t1:.3f}s '
                      f'pin={_t3-_t2:.3f}s fwd={_t4-_t3:.3f}s d2h={_t5-_t4:.3f}s', flush=True)
            if past_key_values is None:
                print(f'  group {_wi} done ({(time.time()-t0):.0f}s)', flush=True)
            # 2026-08-25: empty_cache 显存压力才触发(防OOM), 平时不调保速度
            if torch.cuda.memory_allocated() > 0.85 * torch.cuda.get_device_properties(0).total_memory:
                torch.cuda.empty_cache()

        h = self.hc_head(h)                       # [B,S,D]
        h = F.rms_norm(h, (cfg.hidden_size,), self.norm_w.to(device), cfg.rms_norm_eps)
        logits = h @ self.lm_head_w.T             # lm_head 常驻 GPU
        if return_hidden:
            return logits, cache, h               # h: [B,S,D] 供 MTP 预测下一个 token
        return logits, cache

    # === 生成 (prefill + 增量 decode) ===
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 1048576,
                 do_sample: bool = True, temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9, repetition_penalty: float = 1.15, eos_token_id: int = 1,
                 banned_ids=None):
        generated = input_ids.clone()
        S = input_ids.shape[1]

        # === 流式 prefill：超长输入分段处理（解决 [B,1M,4096] hidden_states 放不下 GPU）===
        # 每段独立过全部层，KV 通过 cache 跨段累积（数学等价于全量 prefill）
        h = None  # 最后一个 token 的 hidden（供 MTP；MTP 禁用时为 None）
        _need_h = self.mtp is not None  # 只有启用 MTP 才返回 hidden（省内存）
        if S > self.prefill_chunk_size:
            n_seg = (S + self.prefill_chunk_size - 1) // self.prefill_chunk_size
            print(f'[stream-prefill] 输入 {S} tokens, 分 {n_seg} 段(每段≤{self.prefill_chunk_size}) 处理...', flush=True)
            cache = None
            t0 = time.time()
            for seg_idx, start in enumerate(range(0, S, self.prefill_chunk_size)):
                end = min(start + self.prefill_chunk_size, S)
                seg = input_ids[:, start:end]
                if _need_h:
                    logits, cache, h = self.forward(seg, past_key_values=cache, return_hidden=True)
                else:
                    logits, cache = self.forward(seg, past_key_values=cache)
                clen = cache.get_seq_length() if cache is not None else end
                print(f'  [stream-prefill] 段{seg_idx+1}/{n_seg} [{start}:{end}] done, '
                      f'cache_len={clen} ({time.time()-t0:.0f}s)', flush=True)
                del seg
                torch.cuda.empty_cache()
            print(f'[stream-prefill] 全部段完成, cache_len={cache.get_seq_length()}', flush=True)
        else:
            if _need_h:
                logits, cache, h = self.forward(generated, return_hidden=True)  # 常规 prefill（单段）
            else:
                logits, cache = self.forward(generated)

        # MTP 加速: 每轮主模型预测 1 token + MTP 预测 1 token → 一次产出 2 token (速度≈×2)
        # 精度: MTP 预测 1 token ~70%, 2-3 token ~50-60% (无条件接受, 质量略降但加速)
        _mtp_extra = 0  # MTP 额外产出的 token 计数
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
            if nxt_id == eos_token_id:
                generated = torch.cat([generated, nxt], dim=-1)
                break

            # MTP 级联预测后续 token (B, C, ...): 复用 mtp.0 逐级递归
            # 一级: A + 主模型hidden h → 预测 B; 二级: B + MTP-1 hidden → 预测 C
            # 注: 权重只有 mtp.0, 二级属 OOD 递归复用, 质量低于一级但速度≈×(mtp_steps+1)
            mtp_ids = []          # 本轮 MTP 额外预测出的 token 列表
            if self.mtp is not None and h is not None:
                try:
                    cur_h, cur_pos, cur_tok = h, cache.get_seq_length() - 1, nxt
                    for _ in range(self.mtp_steps):
                        _pos = torch.tensor([[cur_pos]], device=nxt.device)
                        # 记录 mtp_cache 长度, 供低置信拒绝时回滚 (避免错误B污染MTP自身cache)
                        _mtp_len = self.mtp_cache.get_seq_length()
                        _logits, cur_h = self.mtp(cur_tok, cur_h, _pos, self.mtp_cache)
                        _last = _logits[0, -1, :].clone().float()
                        if banned_ids:
                            for bid in banned_ids:
                                _last[bid] = -float('inf')
                        # 拒绝采样: 算 B 的 softmax 置信度, 低于阈值则拒绝(不 append, 不进主模型cache)
                        _probs = torch.softmax(_last, dim=-1)
                        _id = _last.argmax().item()
                        _conf = _probs[_id].item()
                        if _id == eos_token_id or _conf < self.mtp_min_conf:
                            # 回滚 MTP cache 到本轮调用前 (丢弃错误B的KV)
                            try:
                                for _li in range(len(self.mtp_cache.key_cache)):
                                    self.mtp_cache.key_cache[_li] = self.mtp_cache.key_cache[_li][:, :, :_mtp_len]
                                    self.mtp_cache.value_cache[_li] = self.mtp_cache.value_cache[_li][:, :, :_mtp_len]
                            except Exception:
                                pass
                            break
                        mtp_ids.append(_id)
                        cur_tok = torch.tensor([[_id]], device=nxt.device)
                        cur_pos += 1
                except Exception as _e:
                    print(f'[mtp] 预测失败: {str(_e)[:100]}', flush=True)
                    mtp_ids = []

            if len(mtp_ids) > 0:
                # 批量 prefill: 主模型一次处理 [A, B, C, ...]（2~3 token），比逐次 decode 快
                extra_tok = torch.tensor([mtp_ids], device=nxt.device)
                combined = torch.cat([nxt, extra_tok], dim=-1)  # [1, 1+k]
                generated = torch.cat([generated, combined], dim=-1)
                logits, cache, h = self.forward(combined, past_key_values=cache, return_hidden=True)
                i += combined.shape[1]
                _mtp_extra += len(mtp_ids)
            else:
                # 无 MTP 或 B 是 EOS：主模型只处理 A
                generated = torch.cat([generated, nxt], dim=-1)
                logits, cache, h = self.forward(nxt, past_key_values=cache, return_hidden=True)
                i += 1

            if (i + 1) % 4 == 0:
                print(f'  [gen] token {i+1} done (MTP+{_mtp_extra})', flush=True)
        if self.mtp is not None:
            print(f'[mtp] MTP 加速统计: 额外产出 {_mtp_extra} token', flush=True)
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
