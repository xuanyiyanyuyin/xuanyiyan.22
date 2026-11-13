"""层类型解析 + 常驻权重加载。基于实测 config.json 结构。"""
import json, os, struct, re
import numpy as np
from pathlib import Path
from collections import Counter
from bf16_compat import BF16_DTYPE


def load_layer_types(model_dir: str) -> dict:
    """从 config.json 解析每层注意力类型（处理 text_config 嵌套）"""
    with open(Path(model_dir) / "config.json", "r") as f:
        cfg = json.load(f)

    # Qwen3.5 结构: text_config.layer_types
    tc = cfg.get("text_config", cfg)
    lts = tc.get("layer_types", [])

    if lts:
        layer_types = dict(enumerate(lts))
    else:
        # Fallback: full_attention_interval 模式
        interval = tc.get("full_attention_interval", 4)
        total = tc.get("num_hidden_layers", 60)
        layer_types = {}
        for i in range(total):
            layer_types[i] = "full_attention" if (i + 1) % interval == 0 else "linear_attention"

    print(f"[LayerTypes] {dict(Counter(layer_types.values()))}")
    return layer_types


def load_resident_weights(model_dir: str) -> dict:
    """加载 embed_tokens + lm_head 到 RAM"""
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    with open(idx_path, "r") as f:
        index = json.load(f)

    weights = {}
    for key in ["model.language_model.embed_tokens.weight", "lm_head.weight"]:
        if key not in index["weight_map"]:
            continue
        shard = index["weight_map"][key]
        shard_path = str(Path(model_dir) / shard)

        with open(shard_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        header_end = n + 8

        meta = hdr[key]
        off_s, off_e = meta["data_offsets"]
        shape = tuple(meta["shape"])

        fd = os.open(shard_path, os.O_RDONLY | os.O_BINARY)
        try:
            os.lseek(fd, header_end + off_s, os.SEEK_SET)
            raw = b""
            rem = off_e - off_s
            while rem > 0:
                chunk = os.read(fd, rem)
                if not chunk:
                    break
                raw += chunk
                rem -= len(chunk)
        finally:
            os.close(fd)

        weights[key] = np.frombuffer(raw, dtype=BF16_DTYPE).reshape(shape)
        print(f"[Resident] {key}: {shape}, {weights[key].nbytes/1e6:.0f} MB")

    return weights
