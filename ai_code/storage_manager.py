"""
存储管理器 — 三级存储(GPU/RAM/SSD)调度
不动计算图, 只管理权重的加载/释放时机
"""
import torch
import gc
from collections import OrderedDict


class StorageManager:
    """
    GPU 显存 = 当前计算必需, 算完就丢
    内存 RAM = 下一步要用, 先留着
    """
    def __init__(self, device="cuda", ram_device="cpu"):
        self.device = device
        self.ram_device = ram_device

        # 所有权重常驻内存(CPU RAM), 按名索引
        self.weight_ram: dict[str, torch.Tensor] = {}

        # GPU 上当前的权重(应该很少, 最多一个 Linear 的量)
        self.weight_gpu: OrderedDict[str, torch.Tensor] = OrderedDict()

        # 中间结果暂存内存
        self.results_ram: dict[str, torch.Tensor] = {}

        # 统计
        self.load_count = 0
        self.offload_count = 0
        self.total_bytes_moved = 0
        self.peak_gpu_bytes = 0

    def register_model(self, model):
        """把所有模型权重复制到 CPU RAM, 记录名字"""
        count = 0
        for name, param in model.named_parameters():
            key = self._sanitize(name)
            self.weight_ram[key] = param.data.to(self.ram_device).clone()
            count += 1
            if param.data.device.type == "cuda":
                param.data = param.data.to(self.ram_device)
        print(f"[StorageManager] 注册 {count} 个参数到 RAM, "
              f"总计 {self._ram_usage():.1f} MB")

    def _sanitize(self, name: str) -> str:
        return name.replace(".", "_")

    def _ram_usage(self) -> float:
        total = sum(w.numel() * w.element_size() for w in self.weight_ram.values())
        return total / 1024 / 1024

    def _gpu_usage(self) -> float:
        if not self.weight_gpu:
            return 0
        total = sum(w.numel() * w.element_size() for w in self.weight_gpu.values())
        return total / 1024 / 1024

    def load_to_gpu(self, name: str) -> torch.Tensor:
        """从内存加载一个权重到 GPU, 返回 GPU 上的 tensor"""
        if name in self.weight_gpu:
            return self.weight_gpu[name]

        if name not in self.weight_ram:
            raise KeyError(f"权重 '{name}' 未注册")

        w = self.weight_ram[name].to(self.device)
        self.weight_gpu[name] = w
        self.load_count += 1
        self.total_bytes_moved += w.numel() * w.element_size()

        current = self._gpu_usage()
        if current > self.peak_gpu_bytes:
            self.peak_gpu_bytes = current

        return w

    def offload(self, name: str):
        """从 GPU 释放权重回内存, 但 RAM 副本一直保留"""
        if name not in self.weight_gpu:
            return

        w = self.weight_gpu.pop(name)
        self.total_bytes_moved += w.numel() * w.element_size()
        self.offload_count += 1
        # 不需要复制回 RAM, 因为 RAM 里已经有副本
        del w

    def offload_all(self):
        """释放 GPU 上所有权重"""
        names = list(self.weight_gpu.keys())
        for name in names:
            self.offload(name)

    def keep_result(self, name: str, tensor: torch.Tensor):
        """中间结果暂存内存, 从 GPU 移到 CPU"""
        self.results_ram[name] = tensor.to(self.ram_device)

    def get_result(self, name: str) -> torch.Tensor:
        """从内存取回中间结果到 GPU"""
        if name not in self.results_ram:
            raise KeyError(f"中间结果 '{name}' 不存在")
        return self.results_ram.pop(name).to(self.device)

    def discard_result(self, name: str):
        """丢弃中间结果"""
        if name in self.results_ram:
            del self.results_ram[name]

    def stats(self) -> str:
        return (
            f"[StorageManager] 加载:{self.load_count}次 "
            f"卸载:{self.offload_count}次 "
            f"搬运:{self.total_bytes_moved/1024**3:.2f}GB "
            f"GPU峰值:{self.peak_gpu_bytes/1024**2:.1f}MB "
            f"RAM:{self._ram_usage():.0f}MB "
            f"结果缓存:{len(self.results_ram)}个"
        )

    def cleanup(self):
        """完全清理"""
        self.weight_gpu.clear()
        self.results_ram.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LinearOnDemand:
    """
    替代 nn.Linear 的按需加载版本。
    每次 forward() 时:
      1. 从 RAM 加载权重到 GPU
      2. 计算
      3. 释放权重
      4. 返回结果(调用方负责把结果移到 RAM)
    """
    def __init__(self, name: str, linear: torch.nn.Linear, manager: StorageManager):
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias is not None
        self.manager = manager

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_name = f"{self.name}_weight"
        bias_name = f"{self.name}_bias" if self.bias else None

        # 加载权重
        w = self.manager.load_to_gpu(weight_name)

        # 计算
        if bias_name:
            b = self.manager.load_to_gpu(bias_name)
            result = torch.nn.functional.linear(x, w, b)
            self.manager.offload(bias_name)
        else:
            result = torch.nn.functional.linear(x, w)

        # 释放权重
        self.manager.offload(weight_name)

        return result


class CachedLinearOnDemand:
    """
    带缓存的按需 Linear: 如果权重已经在 GPU 上就不重新加载。
    适用于短时间内多次调用的场景。
    """
    def __init__(self, name: str, linear: torch.nn.Linear, manager: StorageManager):
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias is not None
        self.manager = manager

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_name = f"{self.name}_weight"
        w = self.manager.load_to_gpu(weight_name)

        if self.bias:
            b = self.manager.load_to_gpu(f"{self.name}_bias")
            result = torch.nn.functional.linear(x, w, b)
        else:
            result = torch.nn.functional.linear(x, w)

        return result

    def done(self):
        """调用方用完这个 Linear 后调用, 释放权重"""
        self.manager.offload(f"{self.name}_weight")
        if self.bias:
            self.manager.offload(f"{self.name}_bias")
