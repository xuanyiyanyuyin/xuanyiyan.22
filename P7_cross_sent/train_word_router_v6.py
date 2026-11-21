"""
P7 v6 — Structure-Constrained Word Router
===========================================
v5基础上加3个结构约束loss:
  predicate_keep: 谓语必须保留 (输出向量靠近原动词)
  slot_type:      疑问词只能出现在目标槽位
  question_pos:   疑问槽位的输出必须匹配疑问词

训练数据: data_p7_v3.txt (~350正样本 + 负样本行)
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
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--save_every", type=int, default=2000)
args = parser.parse_args()

LOG_PATH = os.path.join(BASE_DIR, "logs", f"p7v6_{time.strftime('%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")


def w(s):
    log.write(s + "\n"); print(s)


w("=" * 70)
w(f"  P7 v6: Structure-Constrained Word Router")
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
w(f"[P3栈] 5 modules [OK]")

# ── 加载数据 ──
question_words = {"什么", "谁", "哪里", "怎么", "为什么", "哪个", "怎样"}


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) != 2: continue
            A_words = parts[0].split()
            B_words = parts[1].split()
            pairs.append((A_words, B_words))
    return pairs


pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_v3.txt")
w(f"[数据] {len(pairs)}对")

# 分离正负样本 (#NEG标记在注释中, 实际都是正样本格式)
# 负样本行的特点是B句包含错位疑问词或谓语替换
# 我们用P3栈自动检测哪些是负样本
positive_pairs = []
negative_pairs = []
for A_words, B_words in pairs:
    # 负样本检测: 如果B句中的疑问词位置不对, 或者是谓语替换型
    is_neg = False
    A_packets = p3_stack.process_sentence(A_words)
    B_packets = p3_stack.process_sentence(B_words)

    # 找A的谓语位置
    verb_pos_A = -1
    for i, p in enumerate(A_packets):
        if p.best_syntax() == "谓语":
            verb_pos_A = i; break

    # 检查B的谓语位置是否被疑问词替换 (负样本标志)
    if verb_pos_A >= 0 and verb_pos_A < len(B_words):
        if B_words[verb_pos_A] in question_words:
            is_neg = True

    if is_neg:
        negative_pairs.append((A_words, B_words))
    else:
        positive_pairs.append((A_words, B_words))

w(f"  正样本: {len(positive_pairs)} | 负样本: {len(negative_pairs)}")

# 编码正样本
encoded_pos = []
for A_words, B_words in positive_pairs:
    A_vecs = [enc(w) for w in A_words]
    B_vecs = [enc(w) for w in B_words]
    if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
        # P3标注
        A_packets = p3_stack.process_sentence(A_words)
        B_packets = p3_stack.process_sentence(B_words)
        encoded_pos.append((torch.stack(A_vecs), torch.stack(B_vecs),
                            A_words, B_words, A_packets, B_packets))
w(f"[编码] {len(encoded_pos)}对有效正样本")

# 全词池
all_words = set()
for _, B_words in pairs: all_words.update(B_words)
for A_words, _ in pairs: all_words.update(A_words)
vocab_pool = sorted(all_words)
vocab_vecs_list = [enc(w) for w in vocab_pool if enc(w) is not None]
vocab_pool = [w for w, v in zip(vocab_pool, vocab_vecs_list) if v is not None]
vocab_table = F.normalize(torch.stack(vocab_vecs_list), dim=-1)
w(f"[词池] {len(vocab_pool)}词\n")

# ── P7 v6 ──
model = P7WordRouter2048().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
w(f"[P7 v6] {n_params:,} params")

nn.init.normal_(model.q_proj.weight, 0, 0.02)
for proj in [model.k_proj, model.v_proj]:
    nn.init.normal_(proj.weight, 0, 0.02)
    with torch.no_grad():
        proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
nn.init.normal_(model.out_proj.weight, 0, 0.02)
model.temperature.data.fill_(2.0)
model.pos_weight.data.fill_(0.05)

# 从checkpoint恢复
start_epoch = 0
if args.resume and os.path.exists(args.resume):
    resume_ckpt = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    start_epoch = resume_ckpt.get("epoch", 0)
    w(f"[恢复] {args.resume} (epoch {start_epoch})")

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
last_loss = 1.0; total_t0 = time.time()

parser.add_argument("--resume", type=str, default="", help="从checkpoint恢复训练")

# 损失权重: v5基础 + v6结构约束(课程学习)
w_group = 0.5; w_div = 0.8; w_cov = 0.2; w_contrast = 0.3
w_pred_keep_full = 0.8; w_slot_full = 0.5; w_qpos_full = 0.5

# 课程学习: 结构loss从0开始, 位置分化后再加入
STRUCT_WARMUP = 400   # 前400轮不加结构约束, 让v5基础loss建立位置分化
STRUCT_RAMP = 600     # 400-1000轮逐步增加结构约束权重

w(f"  Loss = v5基础 + 结构约束(课程: {STRUCT_WARMUP}轮预热, {STRUCT_RAMP}轮渐变)")
w(f"  满权重: pred_keep={w_pred_keep_full} slot={w_slot_full} qpos={w_qpos_full}")
w(f"{'='*70}\n")

for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
    t0 = time.time()
    em = {"group": 0, "div": 0, "cov": 0, "ctr": 0,
          "pred": 0, "slot": 0, "qpos": 0}; nb = 0

    for A_vecs, B_vecs, A_words, B_words, A_pkts, B_pkts in encoded_pos:
        nA, nB = A_vecs.shape[0], B_vecs.shape[0]
        B_set = set(B_words)

        # 批次模式
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

        # v5 losses
        best_per_A = cos_all[:, :nB].max(dim=-1).values
        group_loss = (1.0 - best_per_A).mean()

        if nA > 1:
            pos_sim = torch.mm(wo_n, wo_n.T)
            mask = 1.0 - torch.eye(nA, device=DEVICE)
            div_loss = (pos_sim * mask).sum() / (nA * (nA - 1))
        else:
            div_loss = torch.tensor(0.0, device=DEVICE)

        best_per_B = cos_all[:, :nB].max(dim=0).values
        cov_loss = (1.0 - best_per_B).mean()

        neg_cands = [w for w in vocab_pool if w not in B_set]
        n_neg = min(10, len(neg_cands))
        if n_neg > 0:
            neg_picked = random.sample(neg_cands, n_neg)
            neg_vecs = torch.stack([vocab_vecs_list[vocab_pool.index(w)] for w in neg_picked])
            neg_sim = torch.mm(wo_n, F.normalize(neg_vecs, dim=-1).T)
            contrast_loss = F.relu(neg_sim - 0.2).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=DEVICE)

        # ── v6 结构约束 losses (课程学习) ──
        # 结构权重: 前STRUCT_WARMUP轮=0, 然后逐步增加到满权重
        if epoch <= STRUCT_WARMUP:
            struct_scale = 0.0
        elif epoch <= STRUCT_WARMUP + STRUCT_RAMP:
            struct_scale = (epoch - STRUCT_WARMUP) / STRUCT_RAMP
        else:
            struct_scale = 1.0

        w_pred_keep = w_pred_keep_full * struct_scale
        w_slot = w_slot_full * struct_scale
        w_qpos = w_qpos_full * struct_scale

        # 1. predicate_keep: 谓语位置必须保留
        pred_loss = torch.tensor(0.0, device=DEVICE)
        pred_count = 0
        for i, pkt in enumerate(A_pkts):
            if pkt.best_syntax() == "谓语" and i < nB:
                # A的谓语位置i → B的相同位置也应该是谓语(相同动词)
                if B_words[i] not in question_words:
                    bv_n = Bv_n[i]
                    cos_val = (wo_n[i] * bv_n).sum()
                    pred_loss = pred_loss + (1.0 - cos_val)
                    pred_count += 1
        if pred_count > 0:
            pred_loss = pred_loss / pred_count

        # 2. slot_type: 疑问词槽位的输出词必须在正确属性族中
        slot_loss = torch.tensor(0.0, device=DEVICE)
        slot_count = 0
        for i, pkt in enumerate(B_pkts):
            if pkt.is_question_word() and i < nA:
                # B位置i是疑问词 → A的输出[i]应该匹配这个疑问词
                bv_n = Bv_n[i]
                cos_val = (wo_n[i] * bv_n).sum()
                slot_loss = slot_loss + (1.0 - cos_val)
                slot_count += 1
        if slot_count > 0:
            slot_loss = slot_loss / slot_count

        # 3. question_pos: 非疑问槽位不能输出疑问词
        qpos_loss = torch.tensor(0.0, device=DEVICE)
        qpos_count = 0
        question_word_ids = []
        for i, pkt in enumerate(B_pkts):
            if pkt.is_question_word():
                question_word_ids.append(i)

        if question_word_ids:
            for i in range(nA):
                if i not in question_word_ids:
                    # 非疑问位置 → 输出不应该靠近疑问词
                    for qid in question_word_ids:
                        if qid < nB:
                            cos_val = (wo_n[i] * Bv_n[qid]).sum()
                            qpos_loss = qpos_loss + F.relu(cos_val - 0.5)
                            qpos_count += 1
        if qpos_count > 0:
            qpos_loss = qpos_loss / qpos_count

        loss = (w_group * group_loss + w_div * div_loss +
                w_cov * cov_loss + w_contrast * contrast_loss +
                w_pred_keep * pred_loss + w_slot * slot_loss + w_qpos * qpos_loss)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        em["group"] += group_loss.item(); em["div"] += div_loss.item()
        em["cov"] += cov_loss.item(); em["ctr"] += contrast_loss.item()
        em["pred"] += pred_loss.item(); em["slot"] += slot_loss.item()
        em["qpos"] += qpos_loss.item(); nb += 1
        last_loss = loss.item()

    for k in em: em[k] /= nb
    elapsed = time.time() - t0

    if epoch % args.display == 0 or epoch == 1:
        total_elapsed = time.time() - total_t0
        eta = total_elapsed / epoch * (args.epochs - epoch)
        w(f"\n{'='*70}")
        w(f"[P7 v6] E{epoch:5d}/{args.epochs} | "
          f"group={em['group']:.4f} div={em['div']:.4f} cov={em['cov']:.4f}")
        w(f"  struct={struct_scale:.2f} pred={em['pred']:.4f} slot={em['slot']:.4f} "
          f"qpos={em['qpos']:.4f} ctr={em['ctr']:.4f}")
        w(f"  GPU={torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB | "
          f"{total_elapsed:.0f}s | ETA {eta/60:.0f}min")

        # 验证谓语保留率
        A_v, B_v, A_w, B_w, A_p, B_p = encoded_pos[0]
        wo, _, _ = model(A_v, B_v, last_loss=0.0)
        wo_n = F.normalize(wo, dim=-1)
        Bv_n = F.normalize(B_v, dim=-1)
        cos_m = torch.mm(wo_n, Bv_n.T)

        # 谓语位置检查
        pred_kept = 0; pred_total = 0
        for i, pkt in enumerate(A_p):
            if pkt.best_syntax() == "谓语" and i < len(B_w):
                pred_total += 1
                if cos_m[i].argmax().item() == i:
                    pred_kept += 1

        pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
        w(f"  A={A_w} -> {pred} (true={B_w})")
        w(f"  谓语保留: {pred_kept}/{pred_total} | unique={len(set(pred))}")

        # 采样
        w(f"  [采样]")
        for A_v, B_v, A_w, B_w, A_p, B_p in encoded_pos[1:4]:
            wo, _, _ = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1)
            Bv_n = F.normalize(B_v, dim=-1)
            cos_m = torch.mm(wo_n, Bv_n.T)
            pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
            # 谓语检查
            pred_ok = 0; pred_t = 0
            for i, pkt in enumerate(A_p):
                if pkt.best_syntax() == "谓语" and i < len(B_w):
                    pred_t += 1
                    if cos_m[i].argmax().item() == i:
                        pred_ok += 1
            w(f"  {A_w} -> {pred}  pred_keep={pred_ok}/{pred_t}")
        w(f"{'='*70}")

    elif epoch % 200 == 0:
        total_elapsed = time.time() - total_t0
        eta = total_elapsed / epoch * (args.epochs - epoch)
        w(f"  E{epoch:5d} | group={em['group']:.4f} pred={em['pred']:.4f} "
          f"slot={em['slot']:.4f} | ETA {eta/3600:.1f}h")

    if epoch % args.save_every == 0:
        snap_path = os.path.join(SAVE_DIR, f"P7v6_e{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, snap_path)

# ── 保存 ──
final_path = os.path.join(SAVE_DIR, "P7v6_final.pt")
torch.save({"model_state_dict": model.state_dict(), "epochs": args.epochs}, final_path)

# ── 结构验收 ──
w(f"\n{'#'*60}")
w(f"  P7 v6 结构验收")
w(f"{'#'*60}")

total_pred_keep = 0; total_pred_all = 0
total_slot_correct = 0; total_slot_all = 0
all_unique = 0; test_count = 0

test_subset = encoded_pos[:30]  # 固定30条QA结构测试

for A_v, B_v, A_w, B_w, A_p, B_p in test_subset:
    wo, _, _ = model(A_v, B_v, last_loss=0.0)
    wo_n = F.normalize(wo, dim=-1)
    Bv_n = F.normalize(B_v, dim=-1)
    cos_m = torch.mm(wo_n, Bv_n.T)
    pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]

    # 谓语保留统计
    for i, pkt in enumerate(A_p):
        if pkt.best_syntax() == "谓语" and i < len(B_w):
            total_pred_all += 1
            if cos_m[i].argmax().item() == i:
                total_pred_keep += 1

    # 疑问槽位统计
    for i, pkt in enumerate(B_p):
        if pkt.is_question_word() and i < len(A_w):
            total_slot_all += 1
            if cos_m[i].argmax().item() == i:
                total_slot_correct += 1

    all_unique += len(set(pred))
    test_count += 1

pred_rate = total_pred_keep / max(total_pred_all, 1)
slot_rate = total_slot_correct / max(total_slot_all, 1)
avg_uniq = all_unique / max(test_count, 1)

w(f"  谓语保留率: {pred_rate:.1%} ({total_pred_keep}/{total_pred_all})")
w(f"  疑问槽位准确率: {slot_rate:.1%} ({total_slot_correct}/{total_slot_all})")
w(f"  平均独特性: {avg_uniq:.1f}")
w(f"  验收: {'PASS' if pred_rate > 0.8 and slot_rate > 0.7 else 'NEED MORE'}")
w(f"{'#'*60}")
w(f"[保存] {final_path}")
w(f"[日志] {LOG_PATH}")
log.close()
