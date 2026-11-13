"""
V22 Stage E 读写头 — 移植到 397B
替换 FullAttention KV Cache + DeltaNet Recurrent State
文件持久化 + Faiss检索 + 三层缓存(GPU/RAM/Disk)
"""
import torch, torch.nn.functional as F, numpy as np, time, os, json, struct, threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ═══════════ MEMORY FILE FORMAT ═══════════
@dataclass
class MemHeader:
    """.mem 文件头部 (~2KB, 快速检索)"""
    feature_vec: np.ndarray   # 256D 检索向量
    weight: float              # 可学习重要性
    timestamp: str             # ISO 8601
    layer_id: int
    mem_type: str              # "kv_cache" or "recurrent"
    tags: List[str]

class MemoryHead:
    """
    GPU(RAM/Disk)三层记忆读写头。
    GPU热层: 最近N条, 直接可用
    RAM温层: LRU缓存, Faiss检索
    Disk冷层: .mem文件, 按需加载

    raw_mode=True: 跳过投影, 直接用原始hidden做余弦检索 (训练前默认)
    raw_mode=False: 用训练的projection矩阵 (训练后自动切换)
    """
    def __init__(self, store_dir: str, max_gpu: int = 200, max_ram: int = 5000,
                 hidden_dim: int = 4096, mem_dim: int = 256, raw_mode: bool = True):
        self.store_dir = store_dir
        self.max_gpu = max_gpu
        self.max_ram = max_ram
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim if not raw_mode else hidden_dim  # raw mode uses full hidden
        self.raw_mode = raw_mode

        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(f"{store_dir}/full_attn", exist_ok=True)
        os.makedirs(f"{store_dir}/delta", exist_ok=True)

        # Projection: raw=identity, proj=trained matrix
        if raw_mode:
            self.proj_write = IdentityProjector(hidden_dim)
            self.proj_query = IdentityProjector(hidden_dim)
        else:
            self.proj_write = Projector(hidden_dim, mem_dim)
            self.proj_query = Projector(hidden_dim, mem_dim)

        # GPU hot layer: direct dict access
        self.gpu_hot = OrderedDict()
        # RAM warm layer: numpy arrays with Faiss index
        self.ram_warm = {}
        self.ram_weights = {}
        self.faiss_index = None  # lazy init
        self.index_keys = []
        # Stats
        self.hits = 0; self.misses = 0; self.writes = 0

    def load_trained_weights(self, W_path, b_path):
        """加载训练好的投影矩阵 → 切换到 proj 模式"""
        import numpy as np
        W = np.load(W_path)
        b = np.load(b_path)
        self.proj_write = Projector(self.hidden_dim, self.mem_dim)
        self.proj_write.w = torch.from_numpy(W.copy())
        self.proj_write.b = torch.from_numpy(b.copy())
        self.proj_query = Projector(self.hidden_dim, self.mem_dim)
        self.proj_query.w = torch.from_numpy(W.copy())
        self.proj_query.b = torch.from_numpy(b.copy())
        self.mem_dim = W.shape[0]
        self.raw_mode = False

        # GPU hot layer: direct dict access
        self.gpu_hot: OrderedDict[str, torch.Tensor] = OrderedDict()

        # RAM warm layer: numpy arrays with Faiss index
        self.ram_warm: Dict[str, np.ndarray] = {}
        self.ram_weights: Dict[str, float] = {}
        self.faiss_index = None  # lazy init
        self.index_keys: List[str] = []

        # Stats
        self.hits = 0; self.misses = 0; self.writes = 0

    # ── WRITE ──
    def write(self, hidden: torch.Tensor, layer_id: int, mem_type: str,
              tags: List[str] = None):
        """
        将hidden_state压缩为记忆向量, 写入三层存储。
        hidden: [B,seq,H] or [B,H] — 先池化
        """
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)  # [B,seq,H] → [B,H]
        else:
            pooled = hidden

        # Project to memory dimension
        mem_vec = self.proj_write(pooled)  # [B, 256]
        mem_vec = F.normalize(mem_vec, dim=-1)  # 球面归一化

        B = mem_vec.shape[0]
        for b in range(B):
            vec = mem_vec[b].detach().cpu().numpy()
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"L{layer_id:02d}_{ts}_{self.writes+b}.mem"

            # Write .mem file
            path = f"{self.store_dir}/{mem_type}/{fname}"
            header = MemHeader(
                feature_vec=vec.astype(np.float32),
                weight=1.0,
                timestamp=ts,
                layer_id=layer_id,
                mem_type=mem_type,
                tags=tags or []
            )
            body_np = pooled[b].detach().float().cpu().numpy()
            self._save_mem(path, header, vec)  # save 256D memory vector

            # GPU hot layer
            key = f"{mem_type}/{fname}"
            self.gpu_hot[key] = mem_vec[b].detach()
            while len(self.gpu_hot) > self.max_gpu:
                self.gpu_hot.popitem(last=False)

            # RAM warm layer
            self.ram_warm[key] = vec.astype(np.float32)
            self.ram_weights[key] = 1.0
            self.index_keys.append(key)
            while len(self.ram_warm) > self.max_ram:
                oldest = self.index_keys.pop(0)
                self.ram_warm.pop(oldest, None)
                self.ram_weights.pop(oldest, None)

        self.writes += B
        if self.writes % 100 == 0:
            self._rebuild_index()

    # ── READ ──
    def read(self, hidden: torch.Tensor, layer_id: int, top_k: int = 4,
             mem_type: str = None) -> Optional[torch.Tensor]:
        """
        检索最相关的历史记忆, 返回融合向量。
        """
        if len(self.ram_warm) == 0:
            return None

        # Query projection
        if hidden.dim() == 3:
            query = hidden.mean(dim=1)
        else:
            query = hidden
        query = self.proj_query(query)  # [B, 256]
        query = F.normalize(query, dim=-1)

        B = query.shape[0]
        results = []
        for b in range(B):
            q = query[b].detach().cpu().numpy().astype(np.float32)

            # Faiss search (or fallback cosine)
            if self.faiss_index is not None:
                try:
                    scores, idxs = self.faiss_index.search(q.reshape(1, -1), top_k)
                    retrieved = []
                    for score, idx in zip(scores[0], idxs[0]):
                        if idx < 0 or idx >= len(self.index_keys):
                            continue
                        key = self.index_keys[idx]
                        weight = self.ram_weights.get(key, 0.01)
                        retrieved.append((key, score * weight))
                except:
                    retrieved = self._cosine_search(q, top_k, layer_id, mem_type)
            else:
                retrieved = self._cosine_search(q, top_k, layer_id, mem_type)

            if not retrieved:
                results.append(None)
                self.misses += 1
                continue

            self.hits += 1
            # Load best match from disk and fuse
            best_key, best_score = retrieved[0]
            mem_data = self._load_body(best_key)
            if mem_data is not None:
                # Return stored memory vector directly (already 256D projected)
                recall_vec = torch.from_numpy(mem_data.copy()).float()
                if recall_vec.device != query.device:
                    recall_vec = recall_vec.to(query.device)
                fused = recall_vec  # 256D memory vector, caller projects back
                results.append(fused)
            else:
                results.append(None)
                self.misses += 1

        if all(r is None for r in results):
            return None
        return torch.stack([r for r in results if r is not None])

    # ── Internal ──
    def _cosine_search(self, query: np.ndarray, top_k: int, layer_id: int,
                       mem_type: str = None) -> List[Tuple[str, float]]:
        scores = []
        for key, vec in self.ram_warm.items():
            if mem_type and mem_type not in key:
                continue
            sim = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec) + 1e-8)
            w = self.ram_weights.get(key, 0.01)
            scores.append((key, float(sim) * w))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _save_mem(self, path: str, header: MemHeader, body: np.ndarray, pos: float = -1.0):
        hdr = json.dumps({
            'vec': header.feature_vec.tolist()[:8],
            'weight': header.weight,
            'ts': header.timestamp,
            'layer': header.layer_id,
            'type': header.mem_type,
            'tags': header.tags,
            'pos': pos
        }).encode()
        hdr_len = struct.pack('<I', len(hdr))
        body_raw = body.astype(np.float32).tobytes()
        with open(path, 'wb') as f:
            f.write(hdr_len)
            f.write(hdr)
            f.write(body_raw)

    # ── FULL-BODY WRITE/READ (完整hidden体, 用于KV重建) ──
    def write_full(self, hidden: torch.Tensor, layer_id: int, mem_type: str,
                   pos: float = -1.0, tags: List[str] = None):
        """存完整池化hidden作为body, 256D特征仅用于检索"""
        if hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden
        mem_vec = F.normalize(self.proj_write(pooled.float()), dim=-1)
        B = mem_vec.shape[0]
        for b in range(B):
            vec = mem_vec[b].detach().cpu().numpy().astype(np.float32)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"L{layer_id:02d}_{ts}_{self.writes+b}.mem"
            path = f"{self.store_dir}/{mem_type}/{fname}"
            header = MemHeader(feature_vec=vec, weight=1.0, timestamp=ts,
                               layer_id=layer_id, mem_type=mem_type, tags=tags or [])
            body_np = pooled[b].detach().float().cpu().numpy()
            self._save_mem(path, header, body_np, pos=pos)   # body=完整hidden
            key = f"{mem_type}/{fname}"
            self.gpu_hot[key] = mem_vec[b].detach()
            while len(self.gpu_hot) > self.max_gpu:
                self.gpu_hot.popitem(last=False)
            self.ram_warm[key] = vec
            self.ram_weights[key] = 1.0
            self.index_keys.append(key)
            while len(self.ram_warm) > self.max_ram:
                oldest = self.index_keys.pop(0)
                self.ram_warm.pop(oldest, None)
                self.ram_weights.pop(oldest, None)
        self.writes += B
        if self.writes % 100 == 0:
            self._rebuild_index()

    def read_full(self, hidden: torch.Tensor, layer_id: int, top_k: int = 4,
                  mem_type: str = None, device=None):
        """检索top_k记忆, 返回 [(完整hidden张量, pos, score), ...]"""
        if len(self.ram_warm) == 0:
            return []
        query = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        q = F.normalize(self.proj_query(query.float()), dim=-1)[0].detach().cpu().numpy().astype(np.float32)
        retrieved = self._cosine_search(q, top_k, layer_id, mem_type)
        out = []
        for key, score in retrieved:
            path = f"{self.store_dir}/{key}"
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                hdr_len = struct.unpack('<I', f.read(4))[0]
                hdr = json.loads(f.read(hdr_len))
                raw = f.read()
            body = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())
            if device is not None:
                body = body.to(device)
            out.append((body, hdr.get('pos', -1.0), score))
        if out:
            self.hits += 1
        else:
            self.misses += 1
        return out

    def _load_body(self, key: str) -> Optional[np.ndarray]:
        path = f"{self.store_dir}/{key}"
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            hdr_len = struct.unpack('<I', f.read(4))[0]
            f.seek(4 + hdr_len)
            raw = f.read()
        return np.frombuffer(raw, dtype=np.float32)

    def _rebuild_index(self):
        try:
            import faiss
            dim = self.mem_dim
            self.faiss_index = faiss.IndexFlatIP(dim)
            if self.index_keys:
                data = np.array([self.ram_warm[k] for k in self.index_keys], dtype=np.float32)
                self.faiss_index.add(data)
        except ImportError:
            pass

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        return (f"[MemoryHead] GPU:{len(self.gpu_hot)} RAM:{len(self.ram_warm)} "
                f"Disk:{self.writes} files | hits:{self.hits} misses:{self.misses} "
                f"({rate:.0f}%)")


class IdentityProjector:
    """恒等投影: 不作任何变换, 直接返回原始向量 (raw模式)"""
    def __init__(self, dim: int):
        self.w = torch.eye(dim) * 0.0  # dummy
        self.b = torch.zeros(dim)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()

class Projector:
    """简单的 Linear 投影 (虚拟权重)"""
    def __init__(self, in_dim: int, out_dim: int):
        self.w = torch.randn(out_dim, in_dim) * 0.02
        self.b = torch.zeros(out_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.w.T.to(x.device) + self.b.to(x.device)
