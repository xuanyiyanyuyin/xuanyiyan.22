"""
离线训练记忆投影矩阵 — PyTorch GPU 加速
策略: PCA-256 → InfoNCE 对比学习 (同session=正, 跨session=负)
运行: D:/ai/python/python.exe train_projector.py [--device auto|cpu|cuda]
      122B 需先关闭。auto=数据＜2000强制CPU, ≥2000+有卡用GPU
输出: D:/ai/memory_122B/projectors/L{id}_W.npy + L{id}_b.npy + recall.json
"""
import torch, torch.nn.functional as F, numpy as np
import struct, json, os, sys, time, argparse
from pathlib import Path

# Parse args before setting DEV
ap = argparse.ArgumentParser()
ap.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
args = ap.parse_args()

MEM_DIR = Path(r'D:/ai/memory_122B')
OUT_DIR = MEM_DIR / 'projectors'
OUT_DIR.mkdir(exist_ok=True)
HIDDEN_DIM = 3072
MEM_DIM = 256
MIN_SAMPLES = 5000       # 低于此跳过训练
CPU_THRESHOLD = 2000      # 低于此强制CPU (数据太少GPU反而慢)
SESSION_GAP_SEC = 600
NCE_BATCH = 256
NCE_EPOCHS = 50
NCE_LR = 0.001
NCE_TEMP = 0.07
VAL_RATIO = 0.1

def get_device(n_samples):
    """auto: 数据<2000 or 无CUDA → CPU, 否则GPU"""
    if args.device == 'cpu':
        return 'cpu'
    if args.device == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    # auto
    if not torch.cuda.is_available():
        return 'cpu'
    if n_samples < CPU_THRESHOLD:
        return 'cpu'
    return 'cuda'


def load_layer(lid):
    """加载某层全部 .mem body"""
    bodies, timestamps = [], []
    d = MEM_DIR / f'L{lid:02d}'
    if not d.exists():
        return None, None
    for root, _, files in os.walk(d):
        for fn in files:
            if not fn.endswith('.mem'):
                continue
            try:
                with open(os.path.join(root, fn), 'rb') as f:
                    hl = struct.unpack('<I', f.read(4))[0]
                    hdr = json.loads(f.read(hl))
                    raw = f.read()
                ts = hdr.get('ts', '')
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                if len(arr) != HIDDEN_DIM:
                    continue
                bodies.append(arr)
                timestamps.append(ts)
            except:
                continue
    if not bodies:
        return None, None
    return np.array(bodies, dtype=np.float32), timestamps


def session_ids(timestamps):
    import datetime
    parsed = []
    for ts in timestamps:
        try:
            parsed.append(datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S').timestamp())
        except:
            parsed.append(0)
    parsed = np.array(parsed)
    order = np.argsort(parsed)
    sess = np.zeros(len(parsed), dtype=np.int32)
    sid = 0
    sess[order[0]] = sid
    for i in range(1, len(order)):
        if parsed[order[i]] - parsed[order[i-1]] > SESSION_GAP_SEC:
            sid += 1
        sess[order[i]] = sid
    return torch.tensor(sess, device='cpu')


def train_layer(bodies, sessions, lid, device):
    """PCA-256 → InfoNCE"""
    n = len(bodies)
    n_val = max(1, int(n * VAL_RATIO))
    idx = torch.randperm(n)
    train_i, val_i = idx[n_val:], idx[:n_val]
    sessions = sessions.to('cpu')

    X = torch.from_numpy(bodies).to(device)  # [N, 3072] f32
    mem_gb = torch.cuda.memory_allocated()/1e9 if device=='cuda' else 0
    print(f'  {n} samples, {sessions.max().item()+1} sessions, device={device} VRAM:{mem_gb:.1f}GB', flush=True)

    # ── PCA via SVD ──
    print('  PCA via SVD...', flush=True)
    mean = X.mean(0, keepdim=True)  # [1, 3072]
    Xc = X - mean
    U, S, V = torch.pca_lowrank(Xc, q=MEM_DIM, center=False, niter=5)
    W_pca = V.T  # [3072, 256]
    bias_pca = -(mean @ W_pca)  # [1, 256]

    # Project all
    with torch.no_grad():
        proj = Xc @ W_pca  # [N, 256]

    # ── InfoNCE ──
    W = W_pca.clone().requires_grad_(True)
    b = bias_pca.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W, b], lr=NCE_LR)

    sessions_dev = sessions.to(device)
    X_train = X[train_i]
    s_train = sessions_dev[train_i]

    print(f'  InfoNCE {NCE_EPOCHS} epochs...', flush=True)
    for epoch in range(NCE_EPOCHS):
        perm = torch.randperm(len(train_i), device=device)
        total_loss = 0.0
        batches = 0
        for s in range(0, len(train_i), NCE_BATCH):
            batch_idx = perm[s:s+NCE_BATCH]
            if len(batch_idx) < 4:
                continue

            xb = X_train[batch_idx]  # [B, 3072]
            zb = xb @ W + b          # [B, 256]
            zb = F.normalize(zb, dim=1)  # [B, 256]

            # Cosine sim matrix
            sim = zb @ zb.T / NCE_TEMP  # [B, B]
            # Positive mask: same session
            sb = s_train[batch_idx].unsqueeze(0)  # [1, B]
            pos_mask = (sb == sb.T)  # [B, B]
            pos_mask.fill_diagonal_(False)

            if pos_mask.sum() == 0:
                continue

            # InfoNCE: -log(sum_pos / sum_all)
            exp_sim = sim.exp()
            pos_sum = (exp_sim * pos_mask.float()).sum(1)  # [B]
            all_sum = exp_sim.sum(1)  # [B]
            # Only compute for anchors that have positives
            has_pos = pos_mask.sum(1) > 0
            if has_pos.sum() == 0:
                continue
            loss = -torch.log(pos_sum[has_pos] / all_sum[has_pos]).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1

        if (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch+1}: loss={total_loss/max(batches,1):.4f}', flush=True)

    # ── Evaluate recall@16 ──
    with torch.no_grad():
        W_final = W.detach()
        b_final = b.detach()
        Z_all = F.normalize(X @ W_final + b_final, dim=1)
        Z_train = Z_all[train_i]
        Z_val = Z_all[val_i]
        s_val = sessions_dev[val_i]

        hits = 0
        for i in range(len(val_i)):
            sims = Z_val[i] @ Z_train.T
            sims[i] = -1 if i < len(Z_train) else sims[i]
            top16 = sims.topk(min(16, len(Z_train))).indices
            if (s_train[top16] == s_val[i]).any():
                hits += 1
        recall = hits / max(len(val_i), 1)

    print(f'    recall@16: {recall:.3f}', flush=True)
    return W_final.cpu().numpy(), b_final.cpu().numpy(), recall


def main():
    print('=' * 60)
    print('Memory Projector Training — PyTorch GPU')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')
    print(f'Min samples/layer: {MIN_SAMPLES}')
    print('=' * 60)

    stats = []
    for lid in range(48):
        bodies, timestamps = load_layer(lid)
        if bodies is None:
            continue
        n = len(bodies)
        if n < MIN_SAMPLES:
            if n > 0:
                print(f'L{lid:02d}: {n} samples (SKIP, need {MIN_SAMPLES})')
            stats.append((lid, n, False, 0))
            continue

        print(f'\n--- L{lid:02d} ---')
        t0 = time.perf_counter()
        sessions = session_ids(timestamps)
        dev = get_device(n)
        W, bias, recall = train_layer(bodies, sessions, lid, dev)

        np.save(OUT_DIR / f'L{lid:02d}_W.npy', W)
        np.save(OUT_DIR / f'L{lid:02d}_b.npy', bias)
        print(f'  Saved ({time.perf_counter()-t0:.0f}s)', flush=True)
        stats.append((lid, n, True, recall))

    print(f'\n{"="*60}')
    trained = sum(1 for _, _, ok, _ in stats if ok)
    skipped = sum(1 for _, _, ok, _ in stats if not ok)
    print(f'Trained: {trained}/{trained+skipped} layers')
    for lid, n, ok, recall in stats:
        tag = f'recall={recall:.3f}' if ok else f'{n} samples (skip)'
        print(f'  L{lid:02d}: {tag}')
    print(f'Output: {OUT_DIR}/')


if __name__ == '__main__':
    main()
