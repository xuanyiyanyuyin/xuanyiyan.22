"""
SafeExpertReader v6: mmap + bytes() copy for correct bf16 reading
No safetensors load (slow), no numpy type games. Just mmap slice → copy bytes → torch bf16.
"""
import os, json, struct, mmap
from pathlib import Path
from typing import Dict, Tuple
import torch

class SafeExpertReader:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._mm: Dict[str, mmap.mmap] = {}
        self._headers: Dict[str, dict] = {}
        self._tensor_locs: Dict[Tuple[int, str], tuple] = {}
        self._build_index()

    def _build_index(self):
        idx_path = self.model_dir / "model.safetensors.index.json"
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_layers = {}
        for name, shard in index["weight_map"].items():
            if "experts" in name and "shared" not in name and "mtp" not in name:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i+1 < len(parts):
                        lid = int(parts[i+1]); break
                else: continue
                proj = parts[-1]
                if proj in ("gate_up_proj", "down_proj"):
                    shard_layers.setdefault(shard, []).append((lid, proj, name))

        shard_set = set(shard_layers.keys())
        for shard in shard_set:
            fp = str(self.model_dir / shard)
            fd = os.open(fp, os.O_RDONLY | os.O_BINARY)
            self._mm[shard] = mmap.mmap(fd, os.fstat(fd).st_size, access=mmap.ACCESS_READ)
            os.close(fd)
            with open(fp, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = {"header_size": n, "tensors": json.loads(f.read(n))}

            for lid, proj, name in shard_layers[shard]:
                meta = self._headers[shard]["tensors"][name]
                self._tensor_locs[(lid, proj)] = (
                    shard, meta["data_offsets"][0], meta["data_offsets"][1],
                    meta["shape"][0] if len(meta["shape"])==3 else 1,
                    meta["shape"][1:]
                )

        print(f"[SafeExpertReader v6] {len(self._tensor_locs)} projections, {len(shard_set)} shards mmap'd")

    def read_expert(self, layer: int, expert_id: int):
        import torch
        shard_gu, off_gu, end_gu, n_gu, shape_gu = self._tensor_locs[(layer, "gate_up_proj")]
        per_gu = (end_gu - off_gu) // n_gu
        abs_gu = self._headers[shard_gu]["header_size"] + 8 + off_gu + expert_id * per_gu
        buf = bytes(self._mm[shard_gu][abs_gu:abs_gu+per_gu])  # bytes copy = magic fix!
        gu = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape_gu).clone()

        shard_dn, off_dn, end_dn, n_dn, shape_dn = self._tensor_locs[(layer, "down_proj")]
        per_dn = (end_dn - off_dn) // n_dn
        abs_dn = self._headers[shard_dn]["header_size"] + 8 + off_dn + expert_id * per_dn
        buf = bytes(self._mm[shard_dn][abs_dn:abs_dn+per_dn])
        dn = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shape_dn).clone()

        return gu, dn

    def close(self):
        for m in self._mm.values():
            try: m.close()
            except: pass
        self._mm.clear()

    def __del__(self):
        self.close()
