# -*- coding: utf-8 -*-
"""CPU 线程检查 — affinity / 系统拓扑 / 线程数对性能的阶梯效应
验证: 进程是否被限制在少数核心 (CherryStudio/宿主限制) → build_pool 多线程灾难
"""
import sys, ctypes, time, threading
sys.stdout.reconfigure(encoding='utf-8')

import sys
def p(*a): print(*a, flush=True)
p('=== CPU 拓扑与 affinity ===')

class SYSTEM_INFO(ctypes.Structure):
    _fields_ = [('wProcessorArchitecture', ctypes.c_ushort), ('wReserved', ctypes.c_ushort),
                ('dwPageSize', ctypes.c_ulong), ('lpMinimumApplicationAddress', ctypes.c_void_p),
                ('lpMaximumApplicationAddress', ctypes.c_void_p),
                ('dwActiveProcessorMask', ctypes.POINTER(ctypes.c_ulong)),
                ('dwNumberOfProcessors', ctypes.c_ulong), ('dwProcessorType', ctypes.c_ulong),
                ('dwAllocationGranularity', ctypes.c_ulong), ('wProcessorLevel', ctypes.c_ushort),
                ('wProcessorRevision', ctypes.c_ushort)]
si = SYSTEM_INFO()
ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
print(f'系统逻辑处理器: {si.dwNumberOfProcessors}', flush=True)

hproc = ctypes.windll.kernel32.GetCurrentProcess()
pmask = ctypes.c_size_t(0); smask = ctypes.c_size_t(0)
ok = ctypes.windll.kernel32.GetProcessAffinityMask(hproc, ctypes.byref(pmask), ctypes.byref(smask))
if ok:
    pbits = bin(pmask.value).count('1'); sbits = bin(smask.value).count('1')
    print(f'系统可用核(mask): {sbits} / 进程实际可用核: {pbits}', flush=True)
    print(f'  进程mask={pmask.value:020b}', flush=True)
    print(f'  系统mask={smask.value:020b}', flush=True)
    if pbits < sbits:
        print(f'  ⚠️ 进程被 affinity 限制! 仅 {pbits}/{sbits} 核可用', flush=True)
    else:
        print(f'  ✓ 进程无 affinity 限制 (全部 {pbits} 核可用)', flush=True)
else:
    print(f'GetProcessAffinityMask 失败', flush=True)

# 逻辑核物理核 (Windows CPU 编号)
def logical_to_physical():
    """遍历逻辑处理器, 识别 P-core/E-core (0=hyperthread, 1=core, 2=package 同级)"""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor')
        n = 0
        while True:
            try:
                winreg.EnumKey(key, n); n += 1
            except OSError:
                break
        print(f'CPU 封装数: {n}', flush=True)
    except Exception as e:
        print(f'CPU 拓扑读取失败: {e}', flush=True)

logical_to_physical()

# === 线程数对性能的阶梯测试 (CPU密集) ===
print('\n=== 线程数扫描 (CPU 密集, 模拟 build/expert 竞争) ===', flush=True)
def busy_work(ms=150):
    t0 = time.time()
    x = 0
    while time.time() - t0 < ms:
        x += 1
    return x

for n in (2, 4, 6, 8, 10, 12, 14, 16, 20):
    # 启动 n 线程跑 CPU 密集, 测总耗时 (理想: 并行, 受限于可用核)
    t0 = time.time()
    threads = [threading.Thread(target=busy_work, args=(200,)) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    dt = time.time() - t0
    # 加速比: 若 n 线程在 P 核上并行, dt 应 ~200ms/P_core_used
    eff_cores = n * 0.2 / dt
    print(f'  {n:2d}线程: {dt*1000:5.0f}ms | 等效并行核数={eff_cores:.1f}', flush=True)
print('=== DONE ===', flush=True)
