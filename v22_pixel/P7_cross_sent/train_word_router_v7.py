"""
P7 v7 — 128头 + 自适应Loss归一化
===================================
每头管一个词向量维度 (128头, head_dim=1)
Q/K/V: 128→128 eye-init → 初始注意力≈余弦
EMA归一化: 每个子loss除以自己的EMA, 公平竞争
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os, sys, random, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.stack import P3AttributeStack
from P7_cross_sent.model import P7WordRouter2048

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--display", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--save_every", type=int, default=2000)
parser.add_argument("--resume", type=str, default="")
args = parser.parse_args()

LOG_PATH = os.path.join(BASE_DIR, "logs", f"p7v7_{time.strftime('%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")


def w(s):
    log.write(s + "\n"); log.flush(); print(s, flush=True)


w("=" * 70)
w(f"  P7 v7: 128头 + EMA-Loss归一化")
w(f"  epochs={args.epochs} display={args.display} lr={args.lr}")
w("=" * 70)

# ── P1 ──
p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
p1_ckpt = torch.load(p1_path, map_location=DEVICE)
p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
p1.load_state_dict(p1_ckpt["model_state_dict"])
p11p = os.path.join(SAVE_DIR, "P11_joint_best.pt")
if os.path.exists(p11p):
    p1.output_proj.load_state_dict(torch.load(p11p, map_location=DEVICE)["p1_proj"])
for p in p1.parameters(): p.requires_grad = False; p1.eval()
c2i = p1_ckpt["char2idx"]


@torch.no_grad()
def enc(w):
    c1 = w[0]; c2 = w[0] if len(w) == 1 else w[1]
    if c1 not in c2i or c2 not in c2i: return None
    return p1(torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE), last_loss=0.0)[0]


# ── P3栈 ──
p3_stack = P3AttributeStack()
question_words = {"什么", "谁", "哪里", "怎么", "为什么", "哪个", "怎样"}

# ── 数据 ──
def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) != 2: continue
            pairs.append((parts[0].split(), parts[1].split()))
    return pairs


pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_v3.txt")
w(f"[数据] {len(pairs)}对")

encoded_pos = []
for A_words, B_words in pairs:
    A_vecs = [enc(w) for w in A_words]
    B_vecs = [enc(w) for w in B_words]
    if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
        A_pkts = p3_stack.process_sentence(A_words)
        B_pkts = p3_stack.process_sentence(B_words)
        encoded_pos.append((torch.stack(A_vecs), torch.stack(B_vecs),
                            A_words, B_words, A_pkts, B_pkts))
w(f"[编码] {len(encoded_pos)}对")

# 全词池
all_words = set()
for _, B_words in pairs: all_words.update(B_words)
for A_words, _ in pairs: all_words.update(A_words)
vocab_pool = sorted(all_words)
vocab_vecs_list = [enc(w) for w in vocab_pool if enc(w) is not None]
vocab_pool = [w for w, v in zip(vocab_pool, vocab_vecs_list) if v is not None]
vocab_table = F.normalize(torch.stack(vocab_vecs_list), dim=-1)
w(f"[词池] {len(vocab_pool)}词\n")

# ── P7 v7: 128头, eye-init ──
model = P7WordRouter2048().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
w(f"[P7 v7] {n_params:,} params | heads=128 | attn_dim=128 | eye-init")

start_epoch = 0
if args.resume and os.path.exists(args.resume):
    resume_ckpt = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    start_epoch = resume_ckpt.get("epoch", 0)
    w(f"[恢复] epoch {start_epoch}")

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
last_loss = 1.0; total_t0 = time.time()

# ── EMA Loss归一化 ──
# 每个子loss维护EMA, 归一化后所有loss在同一量级公平竞争
ema = {}  # {loss_name: ema_value}
ema_decay = 0.99

w(f"{'='*70}")
w(f"  EMA归一化: 每个子loss /= EMA, 公平多目标优化")
w(f"  7 losses: group, diversity, coverage, contrast, pred_keep, slot, qpos")
w(f"{'='*70}\n")

for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
    t0 = time.time()
    raw_losses = {"group": 0, "div": 0, "cov": 0, "ctr": 0,
                  "pred": 0, "slot": 0, "qpos": 0}; nb = 0

    for A_vecs, B_vecs, A_words, B_words, A_pkts, B_pkts in encoded_pos:
        nA, nB = A_vecs.shape[0], B_vecs.shape[0]
        B_set = set(B_words)

        # 批次模式: 30%大候选表, 70% B句词+干扰
        if random.random() < 0.3:
            cands = [w for w in vocab_pool if w not in B_set]
            n_extra = random.randint(25, 45)
            picked = random.sample(cands, min(n_extra, len(cands)))
            extra_vecs = torch.stack([vocab_vecs_list[vocab_pool.index(w)] for w in picked])
            B_with_extra = torch.cat([B_vecs, extra_vecs], dim=0)
        else:
            cands = [w for w in vocab_pool if w not in B_set]
            n_dist = random.randint(2, 4)
            picked = random.sample(cands, min(n_dist, len(cands))) if len(cands) >= n_dist else cands
            extra_vecs = torch.stack([vocab_vecs_list[vocab_pool.index(w)] for w in picked]) if picked else torch.empty(0, 128, device=DEVICE)
            B_with_extra = torch.cat([B_vecs, extra_vecs], dim=0) if extra_vecs.shape[0] > 0 else B_vecs

        nB_total = B_with_extra.shape[0]

        # 前向
        word_out, sent_vec, attn = model(A_vecs, B_with_extra, last_loss=last_loss)
        wo_n = F.normalize(word_out, dim=-1)
        Bv_n = F.normalize(B_with_extra, dim=-1)
        cos_all = torch.mm(wo_n, Bv_n.T)

        # 1. Group: 每A输出匹配B句某词
        best_per_A = cos_all[:, :nB].max(dim=-1).values
        group_loss = (1.0 - best_per_A).mean()

        # 2. Diversity: 输出间余弦必须低
        if nA > 1:
            pos_sim = torch.mm(wo_n, wo_n.T)
            mask = 1.0 - torch.eye(nA, device=DEVICE)
            div_loss = (pos_sim * mask).sum() / (nA * (nA - 1))
        else:
            div_loss = torch.tensor(0.0, device=DEVICE)

        # 3. Coverage: B句每词都要有A输出匹配
        best_per_B = cos_all[:, :nB].max(dim=0).values
        cov_loss = (1.0 - best_per_B).mean()

        # 4. Contrastive push: 推离随机词
        neg_cands = [w for w in vocab_pool if w not in B_set]
        n_neg = min(10, len(neg_cands))
        if n_neg > 0:
            neg_picked = random.sample(neg_cands, n_neg)
            neg_vecs = torch.stack([vocab_vecs_list[vocab_pool.index(w)] for w in neg_picked])
            neg_sim = torch.mm(wo_n, F.normalize(neg_vecs, dim=-1).T)
            contrast_loss = F.relu(neg_sim - 0.2).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=DEVICE)

        # 5. Predicate keep: 谓语位置必须保留
        pred_loss = torch.tensor(0.0, device=DEVICE)
        pred_count = 0
        for i, pkt in enumerate(A_pkts):
            if pkt.best_syntax() == "谓语" and i < nB:
                if B_words[i] not in question_words:
                    pred_loss = pred_loss + (1.0 - (wo_n[i] * Bv_n[i]).sum())
                    pred_count += 1
        if pred_count > 0:
            pred_loss = pred_loss / pred_count

        # 6. Slot type: 疑问槽输出匹配疑问词
        slot_loss = torch.tensor(0.0, device=DEVICE)
        slot_count = 0
        for i, pkt in enumerate(B_pkts):
            if pkt.is_question_word() and i < nA:
                slot_loss = slot_loss + (1.0 - (wo_n[i] * Bv_n[i]).sum())
                slot_count += 1
        if slot_count > 0:
            slot_loss = slot_loss / slot_count

        # 7. Question position: 非疑问位置不输出疑问词
        qpos_loss = torch.tensor(0.0, device=DEVICE)
        qpos_count = 0
        q_ids = [i for i, p in enumerate(B_pkts) if p.is_question_word()]
        if q_ids:
            for i in range(nA):
                if i not in q_ids:
                    for qid in q_ids:
                        if qid < nB:
                            qpos_loss = qpos_loss + F.relu((wo_n[i] * Bv_n[qid]).sum() - 0.5)
                            qpos_count += 1
        if qpos_count > 0:
            qpos_loss = qpos_loss / qpos_count

        # ── EMA自适应归一化 ──
        loss_dict = {
            "group": group_loss, "div": div_loss, "cov": cov_loss, "ctr": contrast_loss,
            "pred": pred_loss, "slot": slot_loss, "qpos": qpos_loss,
        }

        normalized_losses = []
        for name, val in loss_dict.items():
            v = val.detach().item()
            if name not in ema:
                ema[name] = v + 1e-4  # 初始值, 防止除零
            else:
                ema[name] = ema_decay * ema[name] + (1 - ema_decay) * v

            # 归一化: loss / EMA → 所有loss量级≈1
            normalized_losses.append(val / (ema[name] + 1e-6))

            # 记录原始值
            raw_losses[name] += v

        loss = torch.stack(normalized_losses).mean()
        nb += 1

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()

    for k in raw_losses: raw_losses[k] /= nb
    elapsed = time.time() - t0

    if epoch % args.display == 0 or epoch == 1:
        total_elapsed = time.time() - total_t0
        eta = total_elapsed / (epoch - start_epoch) * (start_epoch + args.epochs - epoch)
        w(f"\n{'='*70}")
        w(f"[P7 v7] E{epoch:5d}/{start_epoch+args.epochs} | "
          f"g={raw_losses['group']:.4f} div={raw_losses['div']:.4f} "
          f"cov={raw_losses['cov']:.4f} ctr={raw_losses['ctr']:.4f}")
        w(f"  pred={raw_losses['pred']:.4f} slot={raw_losses['slot']:.4f} "
          f"qpos={raw_losses['qpos']:.4f}")
        if ema:
            ema_str = " ".join(f"{k}={ema[k]:.3f}" for k in sorted(ema.keys()))
            w(f"  EMA: {ema_str}")
        w(f"  GPU={torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB | "
          f"{total_elapsed:.0f}s | ETA {eta/60:.0f}min | "
          f"temp={model.temperature.item():.2f} pos_w={model.pos_weight.item():.3f}")

        # 快速验证
        A_v, B_v, A_w, B_w, A_p, B_p = encoded_pos[0]
        wo, _, _ = model(A_v, B_v, last_loss=0.0)
        wo_n = F.normalize(wo, dim=-1)
        Bv_n = F.normalize(B_v, dim=-1)
        cos_m = torch.mm(wo_n, Bv_n.T)
        pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
        uniq = len(set(pred))

        pred_ok = 0; pred_t = 0
        for i, pkt in enumerate(A_p):
            if pkt.best_syntax() == "谓语" and i < len(B_w):
                pred_t += 1
                if cos_m[i].argmax().item() == i:
                    pred_ok += 1

        w(f"  {A_w} -> {pred} (true={B_w}) unique={uniq} pred_keep={pred_ok}/{pred_t}")

        # 采样2个
        for A_v, B_v, A_w, B_w, A_p, B_p in encoded_pos[1:3]:
            wo, _, _ = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1)
            Bv_n = F.normalize(B_v, dim=-1)
            cos_m = torch.mm(wo_n, Bv_n.T)
            pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
            w(f"  {A_w} -> {pred}")
        w(f"{'='*70}")

    elif epoch % 300 == 0:
        total_elapsed = time.time() - total_t0
        eta = total_elapsed / (epoch - start_epoch) * (start_epoch + args.epochs - epoch)
        w(f"  E{epoch:5d} | g={raw_losses['group']:.4f} div={raw_losses['div']:.4f} "
          f"pred={raw_losses['pred']:.4f} | ETA {eta/3600:.1f}h")

    if epoch % args.save_every == 0:
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "ema": ema},
                   os.path.join(SAVE_DIR, f"P7v7_e{epoch}.pt"))

# ── 保存 ──
final_path = os.path.join(SAVE_DIR, "P7v7_final.pt")
torch.save({"model_state_dict": model.state_dict(), "epochs": args.epochs, "ema": ema}, final_path)

# ── 验收 ──
w(f"\n{'#'*60}")
w(f"  P7 v7 验收 (30条QA)")
w(f"{'#'*60}")

total_pred_keep = 0; total_pred_all = 0
total_slot_correct = 0; total_slot_all = 0
all_unique = 0; test_count = 0

for A_v, B_v, A_w, B_w, A_p, B_p in encoded_pos[:30]:
    wo, _, _ = model(A_v, B_v, last_loss=0.0)
    wo_n = F.normalize(wo, dim=-1)
    Bv_n = F.normalize(B_v, dim=-1)
    cos_m = torch.mm(wo_n, Bv_n.T)
    pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]

    for i, pkt in enumerate(A_p):
        if pkt.best_syntax() == "谓语" and i < len(B_w):
            total_pred_all += 1
            if cos_m[i].argmax().item() == i:
                total_pred_keep += 1
    for i, pkt in enumerate(B_p):
        if pkt.is_question_word() and i < len(A_w):
            total_slot_all += 1
            if cos_m[i].argmax().item() == i:
                total_slot_correct += 1

    all_unique += len(set(pred)); test_count += 1

pred_rate = total_pred_keep / max(total_pred_all, 1)
slot_rate = total_slot_correct / max(total_slot_all, 1)
avg_uniq = all_unique / max(test_count, 1)

w(f"  谓语保留率: {pred_rate:.1%} ({total_pred_keep}/{total_pred_all})")
w(f"  疑问槽位率: {slot_rate:.1%} ({total_slot_correct}/{total_slot_all})")
w(f"  平均独特性: {avg_uniq:.1f}")
w(f"  {'PASS' if pred_rate > 0.7 and avg_uniq > 1.5 else 'NEED MORE'}")
w(f"{'#'*60}")
w(f"[保存] {final_path}")
w(f"[日志] {LOG_PATH}")
log.close()
