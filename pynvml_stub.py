# pynvml stub (AMD ROCm 环境无 NVML；训练脚本只读 GPU 利用率/显存用于日志)
class _Util:
    gpu = 50
    memory = 50
class _Mem:
    total = 48 * 1024**3
    used = 12 * 1024**3
    free = 36 * 1024**3
def nvmlInit(): pass
def nvmlShutdown(): pass
def nvmlDeviceGetHandleByIndex(i): return i
def nvmlDeviceGetCount(): return 1
def nvmlDeviceGetUtilizationRates(h): return _Util()
def nvmlDeviceGetMemoryInfo(h): return _Mem()
def nvmlDeviceGetName(h): return b"AMD Radeon W7900D"
