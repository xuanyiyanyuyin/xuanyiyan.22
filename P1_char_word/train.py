"""
P1 字符→词语 训练脚本
=====================
交互式设置: 训练轮次 / 显示间隔 / 显示模式(均值=区间平均, 点值=最终值)
交叉注意力: 16头×64维, Q=字符, K,V=词语表
Loss: 1 - Pearson_r
日志: GPU显存 / 计算时间 / 调制因子 / 注意力分布
"""
import torch
import torch.nn.functional as F
import time, os, sys, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel


# ==================== 交互式设置 ====================

def get_user_settings():
    """运行前询问训练参数"""
    print(f"\n{'='*50}")
    print(f"P1 字符→词语 训练设置")
    print(f"{'='*50}")

    # 训练轮次
    try:
        s = input(f"  训练轮次 (默认{DEFAULT_EPOCHS}): ").strip()
        epochs = int(s) if s else DEFAULT_EPOCHS
    except ValueError:
        epochs = DEFAULT_EPOCHS

    # 显示间隔
    try:
        s = input(f"  每多少轮显示一次日志 (默认{DISPLAY_INTERVAL}): ").strip()
        disp_interval = int(s) if s else DISPLAY_INTERVAL
    except ValueError:
        disp_interval = DISPLAY_INTERVAL

    # 显示模式
    s = input(f"  显示模式 [average=区间均值 / final=每轮终值] (默认{DISPLAY_MODE}): ").strip().lower()
    disp_mode = s if s in ("average", "final") else DISPLAY_MODE

    # 学习率
    try:
        s = input(f"  学习率 (默认{LEARNING_RATE}): ").strip()
        lr = float(s) if s else LEARNING_RATE
    except ValueError:
        lr = LEARNING_RATE

    # 准确率评估频率
    try:
        s = input(f"  每多少轮评估一次准确率 (默认10, 0=不评估直到最后): ").strip()
        eval_every = int(s) if s else 10
    except ValueError:
        eval_every = 10

    print(f"\n  最终设定: epochs={epochs} | 显示间隔={disp_interval} | 模式={disp_mode}")
    print(f"             学习率={lr} | 评估频率={eval_every}轮")
    print(f"{'='*50}\n")

    return epochs, disp_interval, disp_mode, lr, eval_every


# ==================== 数据 ====================

def load_word_list(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = re.match(r'!([^@]+)@(.)@(.)', line)
            if m:
                entries.append((m.group(1), m.group(2), m.group(3)))
            elif len(line) == 2:
                entries.append((line, line[0], line[1]))
    return entries


def build_vocabs(entries):
    chars = set(); words = []
    for w, c1, c2 in entries:
        chars.add(c1); chars.add(c2); words.append(w)
    char_list = sorted(chars)
    char2idx = {c: i for i, c in enumerate(char_list)}
    idx2char = {i: c for c, i in char2idx.items()}
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return char_list, char2idx, idx2char, word2idx, idx2word


def entries_to_pairs(entries, char2idx):
    return [[char2idx[c1], char2idx[c2]] for _, c1, c2 in entries]


# ==================== 评估 ====================

@torch.no_grad()
def compute_accuracy(model, all_pairs, num_words, device):
    model.eval()
    ref = model.get_all_reference_vectors(device)
    correct_top1 = 0; correct_top3 = 0; total = len(all_pairs)

    for i in range(total):
        p = torch.tensor([all_pairs[i]], device=device)
        pred = model(p, last_loss=0.0)
        sims = F.cosine_similarity(pred, ref)
        idx = torch.argsort(sims, descending=True)
        if idx[0].item() == i: correct_top1 += 1
        if i in idx[:3].tolist(): correct_top3 += 1

    model.train()
    return correct_top1 / total, correct_top3 / total


# ==================== 日志 ====================

def gpu_mem():
    if DEVICE.type != "cuda": return (0, 0, 0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2,
            torch.cuda.memory_reserved(DEVICE)/1024**2,
            torch.cuda.max_memory_allocated(DEVICE)/1024**2)


def log_detailed(epoch, avg_loss, top1, top3, details, sample_char_ids,
                 sample_word, idx2char, idx2word, elapsed):
    """每隔N轮的详细日志(含调制因子/注意力)"""
    print(f"\n{'─'*65}")
    print(f"[详细] Epoch {epoch:4d} | 区间平均Loss: {avg_loss:.6f} | "
          f"Top-1: {top1:.2%} | Top-3: {top3:.2%} | {elapsed:.1f}s")

    if details:
        po = details["pos_out"][0]
        co = details["content_out"][0]
        print(f"  输出位置(8D): [{', '.join(f'{v:.2f}' for v in po.tolist())}]")
        print(f"  输出语义(120D): range[{co.min():.2f},{co.max():.2f}] mean={co.mean():.2f}")

        meta = details["meta_mod"]
        print(f"  元学习调制(128D): range[{meta.min():.2f},{meta.max():.2f}] "
              f"mean={meta.mean():.2f}  TOP5:{torch.topk(meta.abs(), min(5,128)).indices.tolist()}")

        exp = details["explore_mod"]
        print(f"  探索区调制(128D): range[{exp.min():.2f},{exp.max():.2f}] mean={exp.mean():.2f}")

        # 调制效果
        attn_raw = details["attn_out_raw"][0]
        attn_mod = details["modulated_attn"][0]
        shift = (attn_mod - attn_raw).norm().item()
        print(f"  调制注入强度(attn位移L2): {shift:.4f}")

        if "attn_weights" in details:
            aw = details["attn_weights"][0]  # [16, 1, N]
            avg_aw = aw.mean(dim=0).squeeze(0)  # [N]
            top_attn = torch.topk(avg_aw, min(5, len(avg_aw)))
            print(f"  注意力TOP5词: {[(idx2word.get(i.item(),'?'), f'{s.item():.3f}') for i, s in zip(top_attn.indices, top_attn.values)]}")

    a, r, m = gpu_mem()
    print(f"  GPU: alloc={a:.1f}MB res={r:.1f}MB peak={m:.1f}MB")
    print(f"{'─'*65}")


def log_short(epoch, avg_loss, top1, top3, elapsed):
    """每轮简短日志"""
    acc_str = f"Top-1:{top1:.2%} Top-3:{top3:.2%}" if top1 > 0 else ""
    print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.6f} | {acc_str} | {elapsed:.1f}s")


# ==================== 训练 ====================

def pearson_loss(pred, target):
    pm = pred.mean(dim=0, keepdim=True)
    tm = target.mean(dim=0, keepdim=True)
    pc = pred - pm; tc = target - tm
    num = (pc * tc).sum()
    den = torch.sqrt((pc**2).sum() * (tc**2).sum() + PEARSON_EPSILON)
    return 1.0 - num / den


def train():
    epochs, disp_interval, disp_mode, lr, eval_every = get_user_settings()

    print(f"设备: {DEVICE}")
    print(f"交叉注意力: {ATTN_HEADS}头×{ATTN_DIM}维 (每头{ATTN_HEAD_DIM}D)")
    print(f"字符: {POS_DIM}D位置 + {CONTENT_DIM}D语义 = {CHAR_DIM}D | 词向量: {WORD_DIM}D\n")

    # 数据
    entries = load_word_list(WORD_LIST_PATH)
    char_list, char2idx, idx2char, word2idx, idx2word = build_vocabs(entries)
    num_chars = len(char_list); num_words = len(entries)
    all_pairs = entries_to_pairs(entries, char2idx)
    print(f"[数据] {num_words}个词 | {num_chars}个唯一字符")

    # 模型
    model = CharToWordModel(num_chars, num_words).to(DEVICE)
    print(f"[模型] 参数: {sum(p.numel() for p in model.parameters()):,} "
          f"| 词语表: {model.word_table.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # 训练状态
    last_loss = 1.0; best_top1 = 0.0
    interval_losses = []  # 攒区间内的loss
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0; nb = 0
        perm = torch.randperm(num_words)

        for bs in range(0, num_words, BATCH_SIZE):
            be = min(bs + BATCH_SIZE, num_words)
            idxs = perm[bs:be]

            pair_ids = torch.tensor([all_pairs[i] for i in idxs], device=DEVICE)
            word_ids = torch.tensor([i for i in idxs], device=DEVICE)

            t1 = time.time()
            pred, details = model(pair_ids, last_loss=last_loss, return_details=True)
            target = model.get_word_target(word_ids)
            loss = pearson_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_ms = (time.time() - t1) * 1000
            epoch_loss += loss.item(); nb += 1
            last_loss = loss.item()

        avg_loss = epoch_loss / max(nb, 1)
        interval_losses.append(avg_loss)
        elapsed = time.time() - t0

        # 评估准确率
        top1 = top3 = 0.0
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == 1 or epoch == epochs):
            top1, top3 = compute_accuracy(model, all_pairs, num_words, DEVICE)

        # 显示日志
        if epoch % disp_interval == 0 or epoch == 1 or epoch == epochs:
            if disp_mode == "average":
                # 区间均值
                recent = interval_losses[-disp_interval:] if len(interval_losses) >= disp_interval else interval_losses
                show_loss = sum(recent) / len(recent)
            else:
                show_loss = avg_loss

            log_detailed(epoch, show_loss, top1, top3, details,
                         pair_ids[0:1], entries[idxs[0].item()][0], idx2char, idx2word, elapsed)
        else:
            log_short(epoch, avg_loss, top1, top3, elapsed)

        # 保存最佳
        if top1 > best_top1:
            best_top1 = top1
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "char2idx": char2idx, "idx2char": idx2char,
                "word2idx": word2idx, "idx2word": idx2word,
                "num_chars": num_chars, "num_words": num_words,
                "top1": top1, "top3": top3,
            }, os.path.join(SAVE_DIR, "P1_best.pt"))

    # ========== 最终评估 ==========
    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    ft1, ft3 = compute_accuracy(model, all_pairs, num_words, DEVICE)
    print(f"# Training Done | Time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"# Final Top-1: {ft1:.2%} | Top-3: {ft3:.2%} | Best: {best_top1:.2%}")
    status = "[PASS] Target met" if ft1 >= 0.80 else "[FAIL] Need more training"
    print(f"# Check: Top-1 > 80% -> {status}")
    print(f"{'#'*60}")

    torch.save({
        "epoch": epochs, "model_state_dict": model.state_dict(),
        "char2idx": char2idx, "idx2char": idx2char,
        "word2idx": word2idx, "idx2word": idx2word,
        "num_chars": num_chars, "num_words": num_words,
        "top1": ft1, "top3": ft3,
    }, os.path.join(SAVE_DIR, "P1_final.pt"))
    print(f"[保存] {SAVE_DIR}/P1_final.pt")

    return model, char2idx, idx2char, word2idx, idx2word


if __name__ == "__main__":
    train()
