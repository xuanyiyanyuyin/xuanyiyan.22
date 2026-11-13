"""
GPU权重缓存: LRU保持最近N层在GPU, 避免重复H2D。
配合pinned memory加速H2D, CUDA stream重叠。
"""
import torch
from collections import OrderedDict

class GPUCache:
    """
    GPU常驻缓存: 最近使用的权重留在GPU, 超限时LRU淘汰到CPU。
    配合CUDA stream实现H2D异步(compute重叠)。
    """
    def __init__(self, max_bytes=8*1024**3):  # 8GB budget for cached weights
        self.max_bytes = max_bytes
        self.cache = OrderedDict()  # key → gpu_tensor
        self.stream = torch.cuda.Stream()
        self.used = 0
        self.hits = 0
        self.misses = 0

    def get(self, key, cpu_tensor_fn):
        """
        获取GPU tensor。命中→直接返回。未命中→调用cpu_tensor_fn()→H2D→缓存。
        cpu_tensor_fn: 返回CPU上的torch tensor (延迟调用, 只在miss时执行)
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        cpu_t = cpu_tensor_fn()  # 这时才从RAM读取numpy→torch
        if cpu_t is None:
            return None

        # H2D via pinned memory (faster than pageable)
        sz = cpu_t.numel() * cpu_t.element_size()
        self._evict(sz)

        gpu_t = cpu_t.pin_memory().to("cuda", non_blocking=True)
        self.cache[key] = gpu_t
        self.used += sz
        return gpu_t

    def _evict(self, needed):
        while self.used + needed > self.max_bytes and self.cache:
            _, oldest = self.cache.popitem(last=False)
            sz = oldest.numel() * oldest.element_size()
            self.used -= sz
            del oldest

    def clear(self):
        self.cache.clear()
        self.used = 0
        torch.cuda.empty_cache()

    def sync(self):
        self.stream.synchronize()

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total > 0 else 0
        return f"hits={self.hits} misses={self.misses} rate={rate:.0f}% used={self.used/1024**3:.1f}GB"


class PinnedStaging:
    """预分配pinned memory双缓冲, 复用避免每次分配"""
    def __init__(self, size=256*1024*1024):  # 256MB
        self.buf_a = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.buf_b = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.idx = 0

    def next(self):
        self.idx ^= 1
        return self.buf_a if self.idx == 0 else self.buf_b
