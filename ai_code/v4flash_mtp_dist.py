# -*- coding: utf-8 -*-
"""V4Flash MTP 置信度分布测试
手动驱动 MTP decode 循环, 记录每次 MTP 预测 B 的 softmax 置信度。
输出: 置信度累积分布表 + 不同阈值下接受率/加速比 → 用于定最优 mtp_min_conf。
"""
import sys, time, torch, json, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:/ai')
import v4flash_chunked_sched as vs
from transformers import AutoTokenizer

TARGET_TOKENS = 120    # 短上下文(避开32GB内存墙), 够统计分布; 长上下文会导致页缓存归零速度失真
CONF_LOGS = []         # (conf, accepted) 所有 MTP 预测记录

print('[dist] 加载引擎...', flush=True)
eng = vs.V4FlashEngine()
tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)

# 长文 prompt (触发长生成)
prompt = ("请写一篇关于\"人工智能与人类创造力\"的深度长文, "
          "从历史、现状、未来三个角度展开, 包含具体例子和你的思考, 尽量写长一些, 大约2000字。")
ids = tok(prompt, return_tensors='pt').input_ids.to(vs.DEVICE)
print(f'[dist] prompt {ids.shape[1]} token', flush=True)

# 预热: prefill + 几轮 decode (排掉冷启动)
print('[dist] 预热...', flush=True)
eng.forward(ids, return_hidden=True)

# 手动 MTP decode 循环
logits, cache, h = eng.forward(ids, return_hidden=True)
n_gen = 0
t0 = time.time()
mtp_extra = 0
reject = 0
try:
    while n_gen < TARGET_TOKENS:
        last = logits[:, -1, :].clone().float()
        A = last.argmax(-1, keepdim=True)
        if A.item() == 1:  # eos
            break
        # MTP 预测 B
        mtp_id = None
        conf = 0.0
        if eng.mtp is not None and h is not None:
            pos = torch.tensor([[cache.get_seq_length() - 1]], device=vs.DEVICE)
            mtp_logits, _ = eng.mtp(A, h, pos, eng.mtp_cache)
            _ml = mtp_logits[0, -1, :].clone().float()
            _pr = torch.softmax(_ml, dim=-1)
            mtp_id = _ml.argmax().item()
            conf = _pr[mtp_id].item()
        # 模拟 generate 的拒绝采样: 用当前 mtp_min_conf
        if mtp_id is not None and mtp_id != 1 and conf >= eng.mtp_min_conf:
            B = torch.tensor([[mtp_id]], device=vs.DEVICE)
            combined = torch.cat([A, B], dim=-1)
            logits, cache, h = eng.forward(combined, past_key_values=cache, return_hidden=True)
            n_gen += 2
            mtp_extra += 1
            CONF_LOGS.append((conf, True))
        else:
            logits, cache, h = eng.forward(A, past_key_values=cache, return_hidden=True)
            n_gen += 1
            if mtp_id is not None:
                reject += 1
                CONF_LOGS.append((conf, False))
        if n_gen % 50 == 0:
            el = time.time() - t0
            print(f'[dist] {n_gen}/{TARGET_TOKENS} tok, 耗时{el:.0f}s ({n_gen/max(el,0.1):.2f}tok/s), '
                  f'MTP_extra={mtp_extra} reject={reject}', flush=True)
            # 边跑边存, 中断也有数据
            with open(r'D:\ai\mtp_conf_data.json', 'w') as _f:
                json.dump({'confs': [c for c, a in CONF_LOGS], 'n_gen': n_gen,
                           'elapsed': el, 'mtp_extra': mtp_extra, 'reject': reject}, _f)
except Exception as e:
    print(f'[dist] 中断: {str(e)[:150]}', flush=True)
finally:
    el = time.time() - t0
    # 统计
    confs = [c for c, a in CONF_LOGS]
    print(f'\n===== MTP 置信度分布 (共 {len(confs)} 次 MTP 预测) =====', flush=True)
    print(f'生成 {n_gen} token, 耗时 {el:.0f}s, 接受(MTP_extra)={mtp_extra}, 拒绝={reject}', flush=True)
    if confs:
        print(f'置信度: min={min(confs):.3f} max={max(confs):.3f} mean={sum(confs)/len(confs):.3f} median={sorted(confs)[len(confs)//2]:.3f}', flush=True)
        print('\n阈值 | 接受率 | 预估加速(相对纯单token) | 预估加速(相对全接受MTP)', flush=True)
        for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            acc = sum(1 for c in confs if c >= th) / len(confs)
            # 接受时每轮2token, 拒绝时1token; 平均每轮token = 1+acc
            avg_tok = 1 + acc
            speedup_vs_single = avg_tok / 1.0
            speedup_vs_full = avg_tok / 2.0
            print(f'≥{th:.1f}  | {acc*100:5.1f}% | {speedup_vs_single:.2f}x | {speedup_vs_full:.2f}x', flush=True)
        # 直方图
        print('\n置信度直方图:', flush=True)
        hist = [0]*10
        for c in confs:
            hist[min(9, int(c*10))] += 1
        for i, cnt in enumerate(hist):
            bar = '#' * (cnt * 40 // max(1, max(hist)))
            print(f'  {i/10:.1f}-{(i+1)/10:.1f}: {cnt:4d} {bar}', flush=True)
    # 存原始数据
    with open(r'D:\ai\mtp_conf_data.json', 'w') as f:
        json.dump({'confs': confs, 'n_gen': n_gen, 'elapsed': el, 'mtp_extra': mtp_extra, 'reject': reject}, f)
    print('\n原始数据已存: D:\\ai\\mtp_conf_data.json', flush=True)
    print('===== DONE =====', flush=True)
