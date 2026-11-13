import sys, time, torch, psutil
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
t0=time.perf_counter()
print('开始 pread 加载 text_encoder (45G, 绕mmap)...', flush=True)
te = f2.load_text_encoder_pread(f2.MODEL + r'\text_encoder')
print(f'pread加载完成 {time.perf_counter()-t0:.0f}s', flush=True)
print('RAM avail:', round(psutil.virtual_memory().available/1024**3,1), 'G', flush=True)
print('语言层:', len(te.model.language_model.layers), flush=True)
# 存下来备用? 不, 只验证加载
print('PREAD_OK', flush=True)
