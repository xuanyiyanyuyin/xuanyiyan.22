"""
BF16 dtype 安全获取：ml_dtypes 优先，numpy 兜底。
"""
import numpy as np

try:
    from ml_dtypes import bfloat16 as _bfloat16
    BF16_DTYPE = np.dtype(_bfloat16)
except ImportError:
    try:
        BF16_DTYPE = np.dtype("bfloat16")
    except TypeError:
        raise RuntimeError("No bfloat16 support. Run: pip install ml-dtypes")

assert BF16_DTYPE.itemsize == 2
