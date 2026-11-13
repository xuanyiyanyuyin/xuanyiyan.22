"""
VL-30B 真质检引擎 — 手写前向, 照搬 122B 套路
架构: 27-block ViT(1152) → deepstack merger → 48-layer FullAttention MoE(2048)
显存: vision encoder GPU驻留(~1.1GB), emb/lm_head GPU, 专家pread流式, 层权重预取
用法: from vl_scorer import VLScorer; scorer.score_image(path)
"""
import torch, torch.nn.functional as F, time, json, os, sys, struct, ctypes
from pathlib import Path
DEV = 'cuda'
sys.path.insert(0, r'D:\ai')
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception: pass
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading, warnings
warnings.filterwarnings('ignore')

MODEL = r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

# ═══════════ ARCHITECTURE CONSTANTS ═══════════
cfg = json.load(open(MODEL + '/config.json', encoding='utf-8'))
tc = cfg['text_config'] if 'text_config' in cfg else cfg
H = tc['hidden_size']                                              # 2048
V_SIZE = tc['vocab_size']
NH = tc['num_attention_heads']                                     # 32
NKV = tc['num_key_value_heads']                                    # 4
HD = tc['head_dim']                                                # 128
EP = tc['rms_norm_eps']
N_LAYERS = tc['num_hidden_layers']                                 # 48
N_EXPERTS = 128
TOP_K = 8
EXP_INTER = 768
ROPE_DIM = 64
SHARED_INTER = 768

vc = cfg.get('vision_config', {})
VIS_H = vc.get('hidden_size', 1152)
VIS_INTER = vc.get('intermediate_size', 4304)
PATCH_SIZE = 16
VIS_BLOCKS = 27
VIS_NH = 16
VIS_HD = 72
DEEPSTACK_LAYERS = [7, 13, 19, 25]

VIS_START = cfg.get('vision_start_token_id', 151652)
VIS_END = cfg.get('vision_end_token_id', 151653)
IMG_TOKEN = cfg.get('image_token_id', 151655)

# ═══════════ WEIGHT LOADING ═══════════
DTYPE_MAP = {'BF16': torch.bfloat16, 'F32': torch.float32, 'F16': torch.float16,
             'I64': torch.int64, 'BOOL': torch.bool}

def load_keys_pread(path, wanted):
    out = {}
    with open(path, 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hs))
        base = 8 + hs
        for k in wanted:
            meta = header[k]
            o0, o1 = meta['data_offsets']
            f.seek(base + o0)
            buf = bytearray(f.read(o1 - o0))
            out[k] = torch.frombuffer(buf, dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
    return out


class PreadExpertReader:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        idx = json.load(open(model_dir + '/model.safetensors.index.json'))
        self._locs = {}
        self.paths = {}
        for name, shard in idx['weight_map'].items():
            if 'experts' in name and 'shared' not in name:
                parts = name.split('.')
                lid = int(parts[parts.index('layers') + 1])
                proj = 'gate_up_proj' if 'gate_up_proj' in name else 'down_proj'
                self._locs[(lid, proj)] = shard
                self.paths[shard] = model_dir + '/' + shard
        self._hdr = {}
        self._tl = threading.local()
        n_exp = sum(1 for k in self._locs) // 2
        print(f'[ExpertReader] {n_exp} layers, {len(self.paths)} shards')

    def _fh(self, shard):
        hs = getattr(self._tl, 'h', None)
        if hs is None:
            hs = {}
            self._tl.h = hs
        if shard not in hs:
            hs[shard] = open(self.paths[shard], 'rb')
        return hs[shard]

    def _header(self, shard):
        hd = self._hdr.get(shard)
        if hd is None:
            with open(self.paths[shard], 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                hd = (n + 8, json.loads(f.read(n)))
            self._hdr[shard] = hd
        return hd

    def read_expert(self, layer, eid):
        out = []
        for proj in ('gate_up_proj', 'down_proj'):
            shard = self._locs[(layer, proj)]
            base, hdr = self._header(shard)
            meta = hdr[f'model.language_model.layers.{layer}.mlp.experts.{proj}']
            o0, o1 = meta['data_offsets']
            n = meta['shape'][0]
            per = (o1 - o0) // n
            f = self._fh(shard)
            f.seek(base + o0 + eid * per)
            buf = f.read(per)
            out.append(torch.frombuffer(buf, dtype=torch.bfloat16).reshape(meta['shape'][1:]))
        return out[0], out[1]

    def close(self):
        hs = getattr(self._tl, 'h', None)
        if hs:
            for fh in hs.values():
                fh.close()


class VLScorer:
    def __init__(self):
        print('[VLScorer] Loading model...', flush=True)
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

        self._load_vision()
        print(f'  Vision: {len(self.vis_w)} keys on GPU', flush=True)

        self._load_text_non_expert()
        self.expert_reader = PreadExpertReader(MODEL)
        self._build_rope()

        self.exp_cache = OrderedDict()
        self.exp_lock = threading.Lock()
        self.cache_hits = [0]
        self.cache_miss = [0]
        self.EXP_CACHE_MAX = 800

        self.pool = ThreadPoolExecutor(max_workers=12)
        self.copy_stream = torch.cuda.Stream()
        self._pref = {}
        self._pref_lid = -1

        torch.cuda.synchronize()
        print(f'[VLScorer] Ready in {time.perf_counter() - t0:.0f}s', flush=True)

    # ═══ VISION ═══
    def _load_vision(self):
        """加载官方vision model (0.5B) + 从checkpoint装权重"""
        from transformers import Qwen3VLMoeVisionModel
        self.vis_model = Qwen3VLMoeVisionModel.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
        # 从checkpoint提取 visual 权重装入官方模型
        idx = json.load(open(MODEL + '/model.safetensors.index.json'))
        shard_keys = {}
        for name, shard in idx['weight_map'].items():
            if 'visual' in name:
                shard_keys.setdefault(shard, []).append(name)
        vis_w = {}
        for shard, names in shard_keys.items():
            tensors = load_keys_pread(MODEL + '/' + shard, names)
            for k, v in tensors.items():
                vis_w[k.replace('model.visual.', '')] = v
            del tensors
        self.vis_w = {k: v.to(DEV).to(torch.bfloat16) for k, v in vis_w.items()}
        # 覆盖官方模型权重
        with torch.no_grad():
            for k, v in self.vis_model.state_dict().items():
                if k in self.vis_w:
                    v.copy_(self.vis_w[k])
        self.vis_model.to(DEV).eval()

    def _vision_forward(self, pixel_values, grid_thw):
        """
        Qwen3-VL vision encoder — 复用官方vision model (校准锚点)
        pixel_values: [N_patches, 1536] from processor
        grid_thw: [1, 3] → [T, H_grid, W_grid]
        """
        with torch.inference_mode():
            vis_out, _ = self.vis_model(
                pixel_values.to(DEV).to(torch.bfloat16),
                grid_thw.to(DEV))
        # vis_out: [N_out, 2048] → 保持 [1, N_out, 2048]
        return vis_out.unsqueeze(0).to(torch.bfloat16)
        """LEGACY手写版保留参考(弃用)
        """
        # pixel_values [N, C*Tpatch*P*P] where Tpatch=2, P=16, C=3
        # Reshape to [1, 3, T=2, H_img, W_img] for Conv3d
        _, h_grid, w_grid = grid_thw[0].tolist()
        T_PATCH = 2  # processor固定: 1536 = 3 * 2 * 16 * 16
        n_patches = pixel_values.shape[0]
        H_img = h_grid * PATCH_SIZE
        W_img = w_grid * PATCH_SIZE
        # Reshape: [H_grid*W_grid, T*C*P*P] → [H_grid*W_grid, T, C, P, P]
        x = pixel_values.float().view(n_patches, T_PATCH, 3, PATCH_SIZE, PATCH_SIZE)
        # Rearrange to [1, C, T, H_img, W_img]
        x = x.view(h_grid, w_grid, T_PATCH, 3, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(3, 2, 0, 4, 1, 5).contiguous()  # [C, T, H_grid, P, W_grid, P]
        x = x.view(1, 3, T_PATCH, H_img, W_img).to(DEV).to(torch.bfloat16)  # [1,3,2,H,W]

        # Conv3d patch embed: [1, 1152, T_out, H_out, W_out]
        pe_w = self.vis_w['patch_embed.proj.weight']  # [1152, 3, 2, 16, 16]
        pe_b = self.vis_w['patch_embed.proj.bias']     # [1152]
        x = F.conv3d(x, pe_w, pe_b, stride=(2, 16, 16))  # [1, 1152, 1, 16, 16]
        B, C, T_out, H_out, W_out = x.shape
        x = x.squeeze(2)  # [1, 1152, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [1, 256, 1152]  B=1, N=256

        # Position embedding
        pos_w = self.vis_w['pos_embed.weight']  # [2304, 1152]
        pos_ids = torch.arange(N_OUT := H_out * W_out, device=DEV).unsqueeze(0)
        pos_embed = pos_w[pos_ids]  # [1, 256, 1152]
        x = x + pos_embed

        deep_features = []
        for bid in range(VIS_BLOCKS):
            # norm1
            n1w = self.vis_w[f'blocks.{bid}.norm1.weight']
            n1b = self.vis_w[f'blocks.{bid}.norm1.bias']
            normed = F.layer_norm(x.float(), [VIS_H], n1w.float(), n1b.float(), EP).to(torch.bfloat16)

            # QKV attention
            qkv_w = self.vis_w[f'blocks.{bid}.attn.qkv.weight']  # [3456, 1152]
            qkv_b = self.vis_w[f'blocks.{bid}.attn.qkv.bias']
            qkv = F.linear(normed, qkv_w, qkv_b)  # [1, N, 3456]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(1, N_OUT, VIS_NH, VIS_HD).transpose(1, 2)  # [1,16,N,72]
            k = k.view(1, N_OUT, VIS_NH, VIS_HD).transpose(1, 2)
            v = v.view(1, N_OUT, VIS_NH, VIS_HD).transpose(1, 2)

            # Simplified RoPE for vision (2D pos)
            rope_dim_v = 32
            h_ids = torch.arange(H_out, device=DEV).view(1, 1, -1, 1).expand(1, VIS_NH, H_out, W_out).reshape(1, VIS_NH, N_OUT)
            w_ids = torch.arange(W_out, device=DEV).view(1, 1, 1, -1).expand(1, VIS_NH, H_out, W_out).reshape(1, VIS_NH, N_OUT)
            freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim_v, device=DEV).float() / rope_dim_v))
            theta = h_ids.float().unsqueeze(-1) * freq.view(1, 1, 1, -1)
            cos, sin = theta.cos().to(torch.bfloat16), theta.sin().to(torch.bfloat16)
            qr, qp = q[..., :rope_dim_v], q[..., rope_dim_v:]
            kr, kp = k[..., :rope_dim_v], k[..., rope_dim_v:]
            q_rot = torch.cat([qr*cos - qr*sin, qp], dim=-1)
            k_rot = torch.cat([kr*cos - kr*sin, kp], dim=-1)

            attn_out = F.scaled_dot_product_attention(q_rot, k_rot, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, N_OUT, VIS_H)
            proj_w = self.vis_w[f'blocks.{bid}.attn.proj.weight']
            proj_b = self.vis_w[f'blocks.{bid}.attn.proj.bias']
            x = x + F.linear(attn_out.to(torch.bfloat16), proj_w, proj_b)

            # norm2 + MLP
            n2w = self.vis_w[f'blocks.{bid}.norm2.weight']
            n2b = self.vis_w[f'blocks.{bid}.norm2.bias']
            normed2 = F.layer_norm(x.float(), [VIS_H], n2w.float(), n2b.float(), EP).to(torch.bfloat16)
            fc1_w = self.vis_w[f'blocks.{bid}.mlp.linear_fc1.weight']
            fc1_b = self.vis_w[f'blocks.{bid}.mlp.linear_fc1.bias']
            fc2_w = self.vis_w[f'blocks.{bid}.mlp.linear_fc2.weight']
            fc2_b = self.vis_w[f'blocks.{bid}.mlp.linear_fc2.bias']
            mlp_out = F.linear(F.gelu(F.linear(normed2, fc1_w, fc1_b)), fc2_w, fc2_b)
            x = x + mlp_out

            if bid in DEEPSTACK_LAYERS:
                deep_features.append(x.float())

        # Deepstack merger
        merged = torch.cat(deep_features, dim=-1)  # [1, N, 4608]
        # Final merger: norm per 1152-segment, then 4608→2048
        f_nw = self.vis_w['merger.norm.weight']  # [1152]
        f_nb = self.vis_w['merger.norm.bias']
        f_f1w = self.vis_w['merger.linear_fc1.weight']  # [4608, 4608]
        f_f1b = self.vis_w['merger.linear_fc1.bias']
        f_f2w = self.vis_w['merger.linear_fc2.weight']  # [2048, 4608]
        f_f2b = self.vis_w['merger.linear_fc2.bias']

        merged_4d = merged.view(1, N_OUT, 4, 1152)
        merged_normed = F.layer_norm(merged_4d.float(), [1152], f_nw.float(), f_nb.float(), EP)
        merged_flat = merged_normed.reshape(-1, 4608).to(torch.bfloat16)
        h = F.gelu(F.linear(merged_flat, f_f1w, f_f1b))
        h = F.linear(h, f_f2w, f_f2b)
        h = h.view(1, N_OUT, 2048)

        return h.to(torch.bfloat16)

    # ═══ TEXT MODEL ═══
    def _load_text_non_expert(self):
        idx = json.load(open(MODEL + '/model.safetensors.index.json'))
        print('  Loading text non-expert...', end='', flush=True)

        self.layer_ram = [{} for _ in range(N_LAYERS)]
        self.emb = None
        self.lm_head = None
        self.final_norm = None

        shard_keys = {}
        for name, shard in idx['weight_map'].items():
            if 'visual' in name:
                continue
            if '.mlp.experts.' in name:
                continue
            shard_keys.setdefault(shard, []).append(name)

        for shard, names in shard_keys.items():
            tensors = load_keys_pread(MODEL + '/' + shard, names)
            for k, v in tensors.items():
                name = k.replace('model.language_model.', '')
                if 'embed_tokens' in name:
                    self.emb = v
                elif 'lm_head' in name:
                    self.lm_head = v
                elif name == 'norm.weight':
                    self.final_norm = v
                elif 'layers.' in name:
                    p = name.split('.')
                    lid = int(p[p.index('layers') + 1])
                    key = '.'.join(p[p.index('layers') + 2:])
                    self.layer_ram[lid][key] = v
            del tensors

        self.emb = self.emb.to(DEV)
        self.lm_head = self.lm_head.to(DEV)
        self.final_norm = self.final_norm.float().to(DEV)
        torch.cuda.synchronize()
        print(f' done ({sum(1 for d in self.layer_ram if d)} layers)', flush=True)

    def _build_rope(self):
        """VL mrope: 用官方rope类支持任意三维位置"""
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding
        from transformers import AutoConfig
        full_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        self.rope_mod = Qwen3VLMoeTextRotaryEmbedding(full_cfg.text_config).to(DEV)
        self.pos_ids = None  # [3, S] 三维位置, prefill时构建
        self.ROPE_FULL = full_cfg.text_config.head_dim  # 128

    def _rope_for_pos(self, pos_ids):
        """pos_ids: [3, S] → cos/sin [1,1,S,128] (T维, 匹配官方apply_rotary)"""
        xdummy = torch.zeros(1, pos_ids.shape[1], H, device=DEV, dtype=torch.bfloat16)
        with torch.no_grad():
            cos, sin = self.rope_mod(xdummy, pos_ids)
        # cos [3,bs,seq,128] 或 [bs,seq,128]
        if cos.dim() == 4:
            cos = cos[0, 0]  # [seq,128] T维
            sin = sin[0, 0]
        elif cos.dim() == 3:
            cos = cos[0]; sin = sin[0]
        return cos.view(1, 1, -1, self.ROPE_FULL).to(torch.bfloat16), \
               sin.view(1, 1, -1, self.ROPE_FULL).to(torch.bfloat16)

    def _prefetch_layer(self, lid):
        with torch.cuda.stream(self.copy_stream):
            self._pref = {k: v.to(DEV, non_blocking=True) for k, v in self.layer_ram[lid].items()}
        self._pref_lid = lid

    def _get_layer(self, lid):
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        return self._pref

    def _get_expert(self, lid, eid):
        key = (lid, eid)
        with self.exp_lock:
            hit = self.exp_cache.get(key)
            if hit is not None:
                self.exp_cache.move_to_end(key)
                self.cache_hits[0] += 1
                return hit
        g, d = self.expert_reader.read_expert(lid, eid)
        with self.exp_lock:
            self.cache_miss[0] += 1
            self.exp_cache[key] = (g, d)
            while len(self.exp_cache) > self.EXP_CACHE_MAX:
                self.exp_cache.popitem(last=False)
        return g, d

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _full_attn(self, h, gpu, lid, pre, cache, cur_pos):
        B, S, _ = h.shape
        q = gpu.get('self_attn.q_proj.weight')
        k = gpu.get('self_attn.k_proj.weight')
        v = gpu.get('self_attn.v_proj.weight')
        o = gpu.get('self_attn.o_proj.weight')
        qn = gpu.get('self_attn.q_norm.weight')
        kn = gpu.get('self_attn.k_norm.weight')
        if q is None:
            return h * 0.99

        Q = (h @ q.T).view(B, S, NH, HD)
        qa = Q.float()  # VL标准注意力: 无gate分裂, 全128维
        if qn is not None:
            qa = qa * torch.rsqrt(qa.pow(2).mean(-1, keepdim=True) + EP) * qn.float()  # VL: weight直接乘
        qa = qa.to(torch.bfloat16).transpose(1, 2)

        K = (h @ k.T).view(B, S, NKV, HD).float()
        Vm = (h @ v.T).view(B, S, NKV, HD)
        if kn is not None:
            K = K * torch.rsqrt(K.pow(2).mean(-1, keepdim=True) + EP) * kn.float()  # VL: weight直接乘
        K = K.to(torch.bfloat16).transpose(1, 2)
        Vm = Vm.transpose(1, 2)

        # mrope: 用 self.pos_ids 生成三维位置 cos/sin
        if pre:
            pids = self.pos_ids  # [3, S]
        else:
            pids = self.pos_ids[:, -1:].repeat(1, 1) if self.pos_ids is not None else None
            if pids is None:
                pids = torch.tensor([[cur_pos],[0],[0]], device=DEV)
        c, s = self._rope_for_pos(pids)  # [1,1,S,128]
        c_q = c.expand(-1, NH, -1, -1); s_q = s.expand(-1, NH, -1, -1)
        qa = qa * c_q + self._rotate_half(qa) * s_q

        c_k = c.expand(-1, NKV, -1, -1); s_k = s.expand(-1, NKV, -1, -1)
        K = K * c_k + self._rotate_half(K) * s_k

        reps = NH // NKV
        K = K.repeat_interleave(reps, 1)
        Vm = Vm.repeat_interleave(reps, 1)

        kv_key = f'kv_{lid}'
        if not pre:
            kv = cache.get(kv_key)
            if kv is not None:
                K = torch.cat([kv[0], K], 2)
                Vm = torch.cat([kv[1], Vm], 2)
        cache[kv_key] = (K, Vm)

        att = F.scaled_dot_product_attention(qa, K, Vm, is_causal=pre)
        return att.transpose(1, 2).contiguous().view(B, S, NH * HD) @ o.T

    def _moe_forward(self, flat, gpu, lid):
        rw = gpu.get('mlp.gate.weight')
        probs = torch.softmax(flat.float() @ rw.float().T, -1)
        tw, ti = torch.topk(probs, TOP_K, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        uids = ti.unique().tolist()

        out = torch.zeros(flat.shape[0], H, device=DEV, dtype=torch.float32)
        futs = {eid: self.pool.submit(self._get_expert, lid, eid) for eid in uids}
        gs, ds, oks = [], [], []
        for eid in uids:
            try:
                gn, dn = futs[eid].result()
            except:
                continue
            gs.append(gn)
            ds.append(dn)
            oks.append(eid)

        E = len(oks)
        xfer_stream = torch.cuda.Stream()

        if E > 0:
            if flat.shape[0] == 1:
                G = torch.stack(gs)
                D = torch.stack(ds)
                with torch.cuda.stream(xfer_stream):
                    Gd = G.to(DEV, non_blocking=True)
                    Dd = D.to(DEV, non_blocking=True)
            else:
                Gd = [g.to(DEV, non_blocking=True) for g in gs]
                Dd = [d.to(DEV, non_blocking=True) for d in ds]

        sgw = gpu.get('mlp.shared_expert_gate.weight')
        if sgw is not None:
            sg = gpu.get('mlp.shared_expert.gate_proj.weight')
            su = gpu.get('mlp.shared_expert.up_proj.weight')
            sd = gpu.get('mlp.shared_expert.down_proj.weight')
            if sg is not None:
                gv = torch.sigmoid(flat.float() @ sgw.float().T)
                out = out + gv * ((F.silu(flat @ sg.T) * (flat @ su.T)) @ sd.T).float()

        if E > 0:
            if flat.shape[0] == 1:
                torch.cuda.current_stream().wait_stream(xfer_stream)
                # VL gate_up [E,2048,1536]: (e,h,d) 输入h=2048, 输出d=1536
                gateup = torch.einsum('th,ehd->ted', flat, Gd)
                inter = F.silu(gateup[:, :, :EXP_INTER]) * gateup[:, :, EXP_INTER:]
                # Dd 是 [E,768,2048] = (e,i,o)
                eout = torch.einsum('tei,eio->teo', inter, Dd)
                wvec = torch.stack([(tw[0][ti[0] == eid]).sum() for eid in oks]).to(DEV)
                out = out + (eout[0].float() * wvec.unsqueeze(-1)).sum(0, keepdim=True)
            else:
                for j, eid in enumerate(oks):
                    mask = (ti == eid).any(-1)
                    if mask.any():
                        t2 = flat[mask]
                        g, d = Gd[j], Dd[j]
                        gw = (tw[mask].float() * (ti[mask] == eid)).sum(-1)
                        # VL gate_up [2048,1536]: 切列=gate+up; down [768,2048] 不转置
                        eout = ((F.silu(t2 @ g[:, :EXP_INTER]) * (t2 @ g[:, EXP_INTER:])) @ d).float()
                        out[mask] += gw.unsqueeze(-1) * eout

        return out, uids

    # ═══ SCORING ═══
    def score_image(self, image_path: str) -> dict:
        """VL-30B 看图点评"""
        t0 = time.perf_counter()
        img = Image.open(image_path).convert('RGB')
        # VL-30B pos_embed 上限 2304 patch → 缩放图片到可处理尺寸
        MAX_PATCH = 2304
        w0, h0 = img.size
        if (w0//16) * (h0//16) > MAX_PATCH:
            scale = (MAX_PATCH * 256) ** 0.5 / max(w0, h0)  # 保守缩放
            new_w = int(w0 * scale // 16 * 16)
            new_h = int(h0 * scale // 16 * 16)
            new_w = max(new_w, 16); new_h = max(new_h, 16)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f'[vl] resize {w0}x{h0} -> {new_w}x{new_h}', flush=True)

        prompt = (
            "请用中文简单评价这张图片的质量, 从构图、光影、清晰度、色彩四个方面各用一句话点评。"
            "最后给一个 1-10 的综合评分。格式: '评分: X/10'"
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        input_ids = inputs['input_ids'].to(DEV)
        pixel_values = inputs.get('pixel_values')
        grid_thw = inputs.get('image_grid_thw')

        # Vision encoder
        if pixel_values is not None and grid_thw is not None:
            with torch.inference_mode():
                vis_features = self._vision_forward(pixel_values, grid_thw)
        else:
            vis_features = None

        # Build embedding: replace image tokens with vision features
        ids = input_ids[0]
        h = F.embedding(input_ids, self.emb)

        if vis_features is not None:
            vis_mask = (ids == IMG_TOKEN)
            n_vis = vis_mask.sum().item()
            n_vf = vis_features.shape[1]
            vis_positions = torch.where(vis_mask)[0]
            vis_features_flat = vis_features[0]
            for i, pos in enumerate(vis_positions[:min(n_vf, n_vis)]):
                h[0, pos] = vis_features_flat[i]

        S = h.shape[1]
        # ═══ 构建三维 mrope position_ids (官方get_rope_index逻辑) ═══
        spatial_merge = 2  # VL vision_config.spatial_merge_size
        if vis_features is not None and grid_thw is not None:
            t_grid, gh, gw = grid_thw[0].tolist()
            vis_mask2 = (ids == IMG_TOKEN)
            vis_idx = torch.where(vis_mask2)[0]
            n_vis = vis_idx.numel()
            # 视觉位置: llm_grid = 原始grid // spatial_merge (16×16 for 32×32)
            lg_t = t_grid; lg_h = gh // spatial_merge; lg_w = gw // spatial_merge
            v_t = torch.zeros(n_vis, dtype=torch.long, device=DEV)
            v_h = torch.zeros(n_vis, dtype=torch.long, device=DEV)
            v_w = torch.zeros(n_vis, dtype=torch.long, device=DEV)
            # 按官方: h_index=arange(lg_h).view(1,-1,1).expand(lg_t,-1,lg_w).flatten()
            # 视觉token顺序 = vision输出顺序 (每merge单元1个token, 行优先)
            t_i = torch.arange(lg_t).view(-1,1).expand(-1, lg_h*lg_w).flatten()
            h_i = torch.arange(lg_h).view(1,-1,1).expand(lg_t,-1,lg_w).flatten()
            w_i = torch.arange(lg_w).view(1,1,-1).expand(lg_t,lg_h,-1).flatten()
            n_grid_tok = lg_t*lg_h*lg_w
            v_t[:n_grid_tok] = t_i; v_h[:n_grid_tok] = h_i; v_w[:n_grid_tok] = w_i
            # 文本: T=H=W, 从视觉最大位置+1开始 (官方: text_len + st_idx)
            st_idx = 0
            # 视觉段前可能有文本
            text_start = int(max(v_h.max().item(), v_w.max().item()) + 1)
            text_len = S - n_vis
            tt = torch.arange(text_len, device=DEV).view(1,-1).expand(3,-1) + text_start
            t_pos = torch.zeros(S, dtype=torch.long, device=DEV)
            h_pos = torch.zeros(S, dtype=torch.long, device=DEV)
            w_pos = torch.zeros(S, dtype=torch.long, device=DEV)
            t_pos[vis_mask2] = v_t; h_pos[vis_mask2] = v_h; w_pos[vis_mask2] = v_w
            t_pos[~vis_mask2] = tt[0]; h_pos[~vis_mask2] = tt[1]; w_pos[~vis_mask2] = tt[2]
        else:
            t_pos = torch.arange(S, device=DEV)
            h_pos = torch.arange(S, device=DEV)
            w_pos = torch.arange(S, device=DEV)
        self.pos_ids = torch.stack([t_pos, h_pos, w_pos]).to(DEV)  # [3,S]
        cache = {}
        cur_pos = S
        gen_ids = []

        # Prefill
        self._prefetch_layer(0)
        for lid in range(N_LAYERS):
            gpu = self._get_layer(lid)
            if lid + 1 < N_LAYERS:
                self._prefetch_layer(lid + 1)

            inn = gpu.get('input_layernorm.weight')
            hn = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EP)
            if inn is not None:
                hn = hn * (1.0 + inn.float())
            hn = hn.to(torch.bfloat16)

            a_out = self._full_attn(hn, gpu, lid, True, cache, cur_pos)
            h = h + a_out

            pn = gpu.get('post_attention_layernorm.weight')
            hn2 = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EP)
            if pn is not None:
                hn2 = hn2 * (1.0 + pn.float())
            flat = hn2.to(torch.bfloat16).reshape(-1, H)

            out, _ = self._moe_forward(flat, gpu, lid)
            h = h + out.to(torch.bfloat16).reshape_as(h)
            del gpu

        # Sample + decode
        hf = h.float()
        hnorm = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EP) * (1.0 + self.final_norm)
        logits = (hnorm[:, -1:, :].to(torch.bfloat16) @ self.lm_head.T).float()
        nt = int(logits[0, -1].argmax().item())
        gen_ids.append(nt)

        self._prefetch_layer(0)
        for step in range(128):
            if nt == self.tokenizer.eos_token_id:
                break
            # decode: 文本位置 T=H=W 递增
            new_pos = torch.tensor([[cur_pos],[cur_pos],[cur_pos]], device=DEV)
            self.pos_ids = torch.cat([self.pos_ids, new_pos], dim=1)
            h = F.embedding(torch.tensor([[nt]], device=DEV), self.emb)

            for lid in range(N_LAYERS):
                gpu = self._get_layer(lid)
                if lid + 1 < N_LAYERS:
                    self._prefetch_layer(lid + 1)
                else:
                    self._prefetch_layer(0)

                inn = gpu.get('input_layernorm.weight')
                hn = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EP)
                if inn is not None:
                    hn = hn * (1.0 + inn.float())
                hn = hn.to(torch.bfloat16)

                a_out = self._full_attn(hn, gpu, lid, False, cache, cur_pos)
                h = h + a_out

                pn = gpu.get('post_attention_layernorm.weight')
                hn2 = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + EP)
                if pn is not None:
                    hn2 = hn2 * (1.0 + pn.float())
                flat = hn2.to(torch.bfloat16).reshape(-1, H)

                out, _ = self._moe_forward(flat, gpu, lid)
                h = h + out.to(torch.bfloat16).reshape_as(h)
                del gpu

            hf = h.float()
            hnorm = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EP) * (1.0 + self.final_norm)
            logits = (hnorm[:, -1:, :].to(torch.bfloat16) @ self.lm_head.T).float()
            nt = int(logits[0, -1].argmax().item())
            gen_ids.append(nt)
            cur_pos += 1

        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        dt = time.perf_counter() - t0

        import re
        score_match = re.search(r'评分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10', output_text)
        overall = float(score_match.group(1)) if score_match else 5.0

        return {
            'commentary': output_text.strip(),
            'overall': round(overall, 1),
            'passed': bool(overall >= 6.0),
            'source': 'vl-30b',
            'time_s': round(dt, 1),
        }


_scorer = None

def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = VLScorer()
    return _scorer


if __name__ == '__main__':
    scorer = get_scorer()
    path = sys.argv[1] if len(sys.argv) > 1 else r'D:\ai\output\images\_quicktest.png'
    if not os.path.exists(path):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new('RGB', (256, 256), color=(255, 128, 0)).save(path)
    result = scorer.score_image(path)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print(json.dumps(result, ensure_ascii=False, indent=2))
