"""
P7 v8 — 128头 + 层级门控(探索→元学习→主代码) + 分阶段目标达标制
=================================================================
架构: loss→探索区(GELU)→元学习区(门控)→主代码(QKV/word_out)
训练: 3阶段课程 (P1保命→P2路由渐变→P3细规则渐变) + 稀疏惩罚 + 维护权重
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
w(f"  P7 v8: 128头 + EMA归一化 + temp钳位 + 角色/位置约束")
w(f"  epochs={args.epochs} display={args.display} lr={args.lr}")
w("=" * 70)

# ═══════════════════════════════════════
# 加载P1编码器 (frozen) + P11投影
# 所有词通过P1编码为128D向量
# ═══════════════════════════════════════
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


pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_5k.txt")
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
w(f"  分阶段loss + 词级探索区:")
w(f"  P1(E1-400): content+slot+div (保命)")
w(f"  P2(E401-1000): +group+cov+ctr (路由)")
w(f"  P3(E1001+): +role+qpos+pex (细规则)")
w(f"{'='*70}\n")

for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
    t0 = time.time()
    raw_losses = {"group": 0, "div": 0, "cov": 0, "ctr": 0,
                  "content": 0, "slot": 0, "qpos": 0, "role": 0, "pex": 0}; nb = 0

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
        word_out, sent_vec, attn, word_gate, sent_gate = model(A_vecs, B_with_extra, last_loss=last_loss)
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

        # 5. Content keep: A和B中同位置同词的必须保留 (覆盖动词、第三人称、保留宾语)
        content_loss = torch.tensor(0.0, device=DEVICE)
        content_count = 0
        for i in range(min(nA, nB)):
            if A_words[i] == B_words[i]:
                content_loss = content_loss + (1.0 - (wo_n[i] * Bv_n[i]).sum())
                content_count += 1
        if content_count > 0:
            content_loss = content_loss / content_count

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

        # 8. Role transform: 主语位置人称切换 (我↔你, 他→谁)
        role_loss = torch.tensor(0.0, device=DEVICE)
        role_count = 0
        person_map = {"我": "你", "你": "我", "他": "谁", "她": "谁"}
        for i, pkt in enumerate(A_pkts):
            if pkt.best_syntax() == "主语" and i < nB:
                aw = A_words[i]
                if aw in person_map:
                    target_w = person_map[aw]
                    # 找target_w在B中的位置
                    for j, bw in enumerate(B_words):
                        if bw == target_w and j < nB:
                            role_loss = role_loss + (1.0 - (wo_n[i] * Bv_n[j]).sum())
                            role_count += 1
                            break
        if role_count > 0:
            role_loss = role_loss / role_count

        # ══ 9个结构化loss (v8完整版) ══
        # 1-4: v5基础 (group/div/cov/ctr)
        # 5: content_keep (A[i]==B[i]同词保留, 替代旧pred_keep)
        # 6: slot_type (疑问槽位准确性)
        # 7: qpos (非疑问位置禁疑问词)
        # 8: role_transform (我→你, 你→我, 他→谁)
        # 9: pex (非疑问槽位禁输出疑问词)
        pex_count = 0
        # 确定哪些位置允许输出疑问词
        allowed_q_positions = set()
        for i, pkt in enumerate(B_pkts):
            if pkt.is_question_word():
                allowed_q_positions.add(i)

        # 对非允许位置, 输出与任何疑问词的余弦都不能太高
        pex_loss = torch.tensor(0.0, device=DEVICE); pex_count = 0
        for i in range(nA):
            if i not in allowed_q_positions:
                for j, bw in enumerate(B_words):
                    if bw in question_words and j < nB:
                        pex_loss = pex_loss + F.relu((wo_n[i] * Bv_n[j]).sum() - 0.3)
                        pex_count += 1
        if pex_count > 0:
            pex_loss = pex_loss / pex_count

        # ═══════════════════════════════════════
        # 分阶段课程学习 (Curriculum Learning)
        # P1(E1-400): content+slot+div 保命三件套, weight=1.0
        # P1→P2(E401-600): P2渐变ramp 0→1, P1维护 1.0→0.5
        # P2(E601-1000): +group+cov+ctr 路由质量, P2=1.0 P1=0.5
        # P2→P3(E1001-1200): P3渐变ramp 0→1, P1→0.3 P2→0.7
        # P3(E1201+): +role+qpos+pex 细规则, 全部维护权重
        # 关键: 已达标loss不关闭, 降为维护权重防灾难性遗忘
        # ═══════════════════════════════════════

        # P1核心 (始终活跃, 阶段切换后降为维护权重)
        core = {"content": content_loss, "slot": slot_loss, "div": div_loss}
        core_targets = {"content": 0.010, "slot": 0.005, "div": 0.88}

        # P2路由 (E400+渐变加入)
        p2 = {"group": group_loss, "cov": cov_loss, "ctr": contrast_loss}
        p2_targets = {"group": 0.010, "cov": 0.010, "ctr": 0.65}

        # P3细规则 (E1000+渐变加入)
        p3 = {"role": role_loss, "qpos": qpos_loss, "pex": pex_loss}
        p3_targets = {"role": 0.005, "qpos": 0.30, "pex": 0.45}

        # 渐变权重
        if epoch <= 400:
            core_weight = 1.0; p2_weight = 0.0; p3_weight = 0.0
        elif epoch <= 600:
            # P2渐变ramp: 400→600, weight 0→1
            core_weight = 1.0 - 0.5 * (epoch - 400) / 200  # 1.0→0.5 (维护)
            p2_weight = (epoch - 400) / 200  # 0→1
            p3_weight = 0.0
        elif epoch <= 1000:
            core_weight = 0.5; p2_weight = 1.0; p3_weight = 0.0
        elif epoch <= 1200:
            # P3渐变ramp: 1000→1200, weight 0→1
            core_weight = 0.5 - 0.2 * (epoch - 1000) / 200  # 0.5→0.3 (维护)
            p2_weight = 1.0 - 0.3 * (epoch - 1000) / 200    # 1.0→0.7 (维护)
            p3_weight = (epoch - 1000) / 200  # 0→1
        else:
            core_weight = 0.3; p2_weight = 0.7; p3_weight = 1.0

        # smooth_hinge: 达标侧弱维护(0.05x), 超标侧全惩罚(1x)
        def smooth_hinge(val, target):
            diff = val - target
            weight = torch.sigmoid(diff * 10.0)  # smooth 0→1
            return F.softplus(weight * diff + (1 - weight) * diff * 0.05)

        total = torch.tensor(0.0, device=DEVICE)
        for name, val in core.items():
            total = total + core_weight * smooth_hinge(val, core_targets[name])
            raw_losses[name] += val.item()
        if p2_weight > 0:
            for name, val in p2.items():
                total = total + p2_weight * smooth_hinge(val, p2_targets[name])
                raw_losses[name] += val.item()
        if p3_weight > 0:
            for name, val in p3.items():
                total = total + p3_weight * smooth_hinge(val, p3_targets[name])
                raw_losses[name] += val.item()
        # 未激活的loss记录0
        for n in ["group","content","slot","role","cov","div","ctr","qpos","pex"]:
            if n not in raw_losses or raw_losses[n] == 0:
                raw_losses[n] = raw_losses.get(n, 0)

        # 稀疏惩罚 + 温控软上限
        sparsity_loss = 0.02 * (word_gate.mean() + sent_gate.mean())
        temp_penalty = 0.01 * F.softplus(model.temperature - 3.0)
        total = total + sparsity_loss + temp_penalty

        total = total / (1 + sum([core_weight] +
                                 ([p2_weight] if p2_weight > 0 else []) +
                                 ([p3_weight] if p3_weight > 0 else [])))

        nb += 1

        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = total.item()

    for k in raw_losses: raw_losses[k] /= nb
    elapsed = time.time() - t0

    if epoch % args.display == 0 or epoch == 1:
        total_elapsed = time.time() - total_t0
        eta = total_elapsed / (epoch - start_epoch) * (start_epoch + args.epochs - epoch)
        if epoch <= 400: phase = "P1(保命)"
        elif epoch <= 600: phase = "P1->P2(渐变)"
        elif epoch <= 1000: phase = "P2(路由)"
        elif epoch <= 1200: phase = "P2->P3(渐变)"
        else: phase = "P3(细规则)"
        w(f"\n{'='*70}")
        w(f"[P7 v8] E{epoch:5d}/{start_epoch+args.epochs} {phase} | "
          f"g={raw_losses['group']:.4f} div={raw_losses['div']:.4f} "
          f"cov={raw_losses['cov']:.4f} ctr={raw_losses['ctr']:.4f}")
        w(f"  content={raw_losses['content']:.4f} slot={raw_losses['slot']:.4f} "
          f"qpos={raw_losses['qpos']:.4f}")
        w(f"  role={raw_losses['role']:.4f} pex={raw_losses['pex']:.4f} "
          f"| temp={min(model.temperature.item(), 3.0):.2f}")
        # 层级门控状态: 探索→元学习→主代码
        lf = min(last_loss*20., 1.)
        lt = torch.tensor([[lf]], device=DEVICE)
        es = model.explore_net(lt) + model.explore_state * lf  # 探索信号
        mi = torch.cat([model.meta_state.unsqueeze(0)*lf, es], dim=-1)
        wg = torch.sigmoid(model.meta_word_gate(mi) * model.gate_sharpness).squeeze()
        sg = torch.sigmoid(model.meta_sent_gate(mi) * model.gate_sharpness).squeeze()
        w(f"  [探索] signal_norm={es.norm().item():.2f}")
        w(f"  [元→词gate] open={(wg<0.5).sum().item()}维 locked={(wg>0.9).sum().item()}维")
        w(f"  [元→句gate] open={(sg<0.5).sum().item()}维 locked={(sg>0.9).sum().item()}维")
        # 超标检测
        all_targets = {**core_targets, **p2_targets, **p3_targets}
        excess_list = []
        for name in sorted(all_targets.keys()):
            actual = raw_losses.get(name, 0)
            target = all_targets[name]
            if actual > target:
                excess_list.append(f"{name}={actual:.3f}>{target:.3f}")
        if excess_list:
            w(f"  EXCESS: {' | '.join(excess_list[:6])}")
        w(f"  weights: core={core_weight:.1f} p2={p2_weight:.1f} p3={p3_weight:.1f}")
        w(f"  GPU={torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB | "
          f"{total_elapsed:.0f}s | ETA {eta/60:.0f}min | "
          f"temp={model.temperature.item():.2f} pos_w={model.pos_weight.item():.3f}")

        # 快速验证
        A_v, B_v, A_w, B_w, A_p, B_p = encoded_pos[0]
        wo, _, _, _, _ = model(A_v, B_v, last_loss=0.0)
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

        w(f"  {A_w} -> {pred} (true={B_w}) unique={uniq} content_keep={pred_ok}/{pred_t}")

        # 采样2个
        for A_v, B_v, A_w, B_w, A_p, B_p in encoded_pos[1:3]:
            wo, _, _, _, _ = model(A_v, B_v, last_loss=0.0)
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
          f"content={raw_losses['content']:.4f} | ETA {eta/3600:.1f}h")

    if epoch % args.save_every == 0:
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "ema": ema},
                   os.path.join(SAVE_DIR, f"P7v8_e{epoch}.pt"))

# ── 保存 ──
final_path = os.path.join(SAVE_DIR, "P7v8_final.pt")
torch.save({"model_state_dict": model.state_dict(), "epochs": args.epochs, "ema": ema}, final_path)

# ═══════════════════════════════════════
# 最终验收: 30条固定QA结构测试
# 指标: 内容保留率(A[i]==B[i]), 疑问槽位准确率, 平均独特性
# ═══════════════════════════════════════
w(f"\n{'#'*60}")
w(f"  P7 v8 验收 (30条QA)")
w(f"{'#'*60}")

total_pred_keep = 0; total_pred_all = 0
total_slot_correct = 0; total_slot_all = 0
all_unique = 0; test_count = 0

for A_v, B_v, A_w, B_w, A_p, B_p in encoded_pos[:30]:
    wo, _, _, _, _ = model(A_v, B_v, last_loss=0.0)
    wo_n = F.normalize(wo, dim=-1)
    Bv_n = F.normalize(B_v, dim=-1)
    cos_m = torch.mm(wo_n, Bv_n.T)
    pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]

    # 内容保留: A[i]==B[i]同位置同词
    for i in range(min(len(A_w), len(B_w))):
        if A_w[i] == B_w[i]:
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

w(f"  内容保留率(A[i]==B[i]): {pred_rate:.1%} ({total_pred_keep}/{total_pred_all})")
w(f"  疑问槽位率: {slot_rate:.1%} ({total_slot_correct}/{total_slot_all})")
w(f"  平均独特性: {avg_uniq:.1f}")
w(f"  温度: {model.temperature.item():.2f} (clamped={min(model.temperature.item(), 3.0):.2f})")
w(f"  {'PASS' if pred_rate > 0.7 and avg_uniq > 1.5 else 'NEED MORE'}")
w(f"{'#'*60}")
w(f"[保存] {final_path}")
w(f"[日志] {LOG_PATH}")
log.close()
