"""
NonExpertStreamer: 每层 2-3 次大块顺序读拉完 238MB 非专家权重。
"""
import os, json, struct, re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
from bf16_compat import BF16_DTYPE


SAFETENSORS_DTYPE_MAP = {
    "F32": np.float32, "F16": np.float16, "BF16": BF16_DTYPE,
    "F64": np.float64, "I64": np.int64, "I32": np.int32,
    "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": bool,
}

@dataclass
class TensorMeta:
    name: str
    local_offset: int
    size: int
    shape: tuple
    np_dtype: np.dtype  # actual dtype from safetensors header

@dataclass
class LayerShardRead:
    shard_path: str
    abs_read_start: int
    read_size: int
    tensors: List[TensorMeta]

@dataclass
class LayerPlan:
    layer_id: int
    shard_reads: List[LayerShardRead]
    total_bytes: int

class NonExpertStreamer:
    def __init__(self, model_dir: str, num_layers: int = 60):
        self.model_dir = Path(model_dir)
        self.num_layers = num_layers
        self._fd_cache: Dict[str, int] = {}
        self._plans: Dict[int, LayerPlan] = {}
        self._build_plans()

    def _build_plans(self):
        idx_path = self.model_dir / "model.safetensors.index.json"
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        layer_tensors: Dict[int, List[str]] = defaultdict(list)
        for name, shard in index["weight_map"].items():
            if "experts." in name and "shared_expert" not in name:
                continue
            m = re.search(r"layers\.(\d+)", name)
            if m:
                layer_tensors[int(m.group(1))].append(name)

        needed_shards = set()
        for names in layer_tensors.values():
            for name in names:
                needed_shards.add(index["weight_map"][name])

        shard_headers: Dict[str, Tuple[dict, int]] = {}
        for shard in needed_shards:
            shard_path = str(self.model_dir / shard)
            with open(shard_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            shard_headers[shard] = (hdr, n + 8)

        for layer_id in range(self.num_layers):
            tensors = layer_tensors.get(layer_id, [])
            if not tensors:
                continue

            shard_groups: Dict[str, List[str]] = defaultdict(list)
            for name in tensors:
                shard_groups[index["weight_map"][name]].append(name)

            shard_reads = []
            total = 0

            for shard, names in shard_groups.items():
                hdr, header_end = shard_headers[shard]
                entries = []
                min_off = float("inf")
                max_end = 0

                for name in names:
                    meta = hdr[name]
                    off_start, off_end = meta["data_offsets"]
                    shape = tuple(meta["shape"])
                    st_dtype = meta.get("dtype", "BF16")
                    np_dt = SAFETENSORS_DTYPE_MAP.get(st_dtype, BF16_DTYPE)
                    entries.append((name, off_start, off_end - off_start, shape, np_dt))
                    min_off = min(min_off, off_start)
                    max_end = max(max_end, off_end)

                read_size = max_end - min_off
                tensor_metas = [
                    TensorMeta(name=n, local_offset=o - min_off, size=s, shape=sh, np_dtype=dt)
                    for n, o, s, sh, dt in entries
                ]
                shard_reads.append(LayerShardRead(
                    shard_path=str(self.model_dir / shard),
                    abs_read_start=header_end + int(min_off),
                    read_size=read_size,
                    tensors=tensor_metas,
                ))
                total += read_size

            self._plans[layer_id] = LayerPlan(
                layer_id=layer_id, shard_reads=shard_reads, total_bytes=total
            )

        total_gb = sum(p.total_bytes for p in self._plans.values()) / 1e9
        print(f"[NonExpertStreamer] {self.num_layers} layers, {total_gb:.2f} GB, "
              f"{total_gb/self.num_layers*1000:.0f} MB/layer")

    def _get_fd(self, path: str) -> int:
        if path not in self._fd_cache:
            self._fd_cache[path] = os.open(path, os.O_RDONLY | os.O_BINARY)
        return self._fd_cache[path]

    def _safe_read(self, fd: int, size: int, offset: int) -> bytes:
        os.lseek(fd, offset, os.SEEK_SET)
        buf = bytearray(size)
        mv = memoryview(buf)
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                raise IOError(f"EOF at offset {offset + size - remaining}")
            n = len(chunk)
            mv[size - remaining:size - remaining + n] = chunk
            remaining -= n
        return bytes(buf)

    def read_layer(self, layer_id: int) -> Dict[str, np.ndarray]:
        plan = self._plans[layer_id]
        result = {}
        for sr in plan.shard_reads:
            fd = self._get_fd(sr.shard_path)
            raw = self._safe_read(fd, sr.read_size, sr.abs_read_start)
            for tm in sr.tensors:
                slice_data = raw[tm.local_offset:tm.local_offset + tm.size]
                dt = tm.np_dtype
                arr = np.frombuffer(slice_data, dtype=dt).reshape(tm.shape)
                result[tm.name] = arr
        return result

    def close(self):
        for fd in self._fd_cache.values():
            os.close(fd)
        self._fd_cache.clear()
