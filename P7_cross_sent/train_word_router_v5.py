"""
P7 2048头词级路由 训练 v5 — 对比推远 + 混合候选表
===================================================
v4问题: 输出向量在公共词附近聚类, 全候选表时所有输入产生相同答案
v5修复: 1) 对比推远loss: 随机负样本cos < 0.3 的margin惩罚
        2) 30%批次用大候选表(30-50词)训练直接检索能力
        3) 加法位置编码(内容主导)
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os, sys, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P7_cross_sent.model import P7WordRouter2048


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


def train():
    ep = 500; disp = 25; lr = 0.001
    print(f"P7-2048 v5 对比推远 epochs={ep} lr={lr}\n")

    # ── P1 ──
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    @torch.no_grad()
    def enc(w):
        c1 = w[0]; c2 = w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    # ── 数据 ──
    pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_v2.txt")
    encoded_pairs = []
    for A_words, B_words in pairs:
        A_vecs = [enc(w) for w in A_words]
        B_vecs = [enc(w) for w in B_words]
        if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
            encoded_pairs.append((torch.stack(A_vecs), torch.stack(B_vecs),
                                  A_words, B_words))
    print(f"[数据] {len(encoded_pairs)}对有效")

    # 全词表(干扰/负样本池)
    all_words = set()
    for _, B_words in pairs: all_words.update(B_words)
    for A_words, _ in pairs: all_words.update(A_words)
    vocab_pool = sorted(all_words)
    vocab_vecs = [enc(w) for w in vocab_pool if enc(w) is not None]
    vocab_pool = [w for w, v in zip(vocab_pool, vocab_vecs) if v is not None]
    vocab_table = F.normalize(torch.stack(vocab_vecs), dim=-1)
    print(f"[词池] {len(vocab_pool)}词\n")

    # ── 模型 ──
    model = P7WordRouter2048().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7-2048 v5] 参数: {n_params:,}")

    nn.init.normal_(model.q_proj.weight, 0, 0.02)
    for proj in [model.k_proj, model.v_proj]:
        nn.init.normal_(proj.weight, 0, 0.02)
        with torch.no_grad():
            proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
    nn.init.normal_(model.out_proj.weight, 0, 0.02)
    model.temperature.data.fill_(2.0)
    model.pos_weight.data.fill_(0.05)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    last_loss = 1.0

    w_group = 0.5; w_div = 0.4; w_cov = 0.2; w_contrast = 0.3; w_sent = 0.05

    print(f"{'='*70}")
    print(f"  Loss = {w_group}*group + {w_div}*diversity + {w_cov}*coverage")
    print(f"       + {w_contrast}*contrastive_push + {w_sent}*sentence")
    print(f"  策略: 70% B句词+2-4干扰 | 30% 大候选表(30-50词)")
    print(f"{'='*70}\n")

    for epoch in range(1, ep + 1):
        t0 = time.time()
        em = {"group": 0, "div": 0, "cov": 0, "contrast": 0, "sent": 0}; nb = 0

        for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
            nA, nB_orig = A_vecs.shape[0], B_vecs.shape[0]

            # ── 决定批次模式 ──
            use_large_table = random.random() < 0.3

            if use_large_table:
                # 模式A: 大候选表 (30-50随机词 + 正确B词)
                B_set = set(B_words)
                cands = [w for w in vocab_pool if w not in B_set]
                n_extra = random.randint(25, 45)
                if len(cands) >= n_extra:
                    picked = random.sample(cands, n_extra)
                else:
                    picked = cands
                extra_vecs = torch.stack([vocab_vecs[vocab_pool.index(w)] for w in picked])
                B_with_extra = torch.cat([B_vecs, extra_vecs], dim=0)
                nB_total = B_with_extra.shape[0]
            else:
                # 模式B: B句词 + 少量干扰
                B_set = set(B_words)
                cands = [w for w in vocab_pool if w not in B_set]
                n_dist = random.randint(2, 4)
                if len(cands) >= n_dist:
                    picked = random.sample(cands, n_dist)
                else:
                    picked = cands[:max(0, len(cands))]
                extra_vecs = torch.stack([vocab_vecs[vocab_pool.index(w)] for w in picked]) if picked else torch.empty(0, 128, device=DEVICE)
                B_with_extra = torch.cat([B_vecs, extra_vecs], dim=0) if extra_vecs.shape[0] > 0 else B_vecs
                nB_total = B_with_extra.shape[0]

            # 前向
            word_out, sent_vec, attn = model(A_vecs, B_with_extra, last_loss=last_loss)
            wo_n = F.normalize(word_out, dim=-1)
            Bv_n = F.normalize(B_with_extra, dim=-1)
            cos_all = torch.mm(wo_n, Bv_n.T)  # [nA, nB_total]

            # 1. Group: 每个A输出匹配B句某词
            best_per_A = cos_all[:, :nB_orig].max(dim=-1).values
            group_loss = (1.0 - best_per_A).mean()

            # 2. Diversity: 输出之间必须不同
            if nA > 1:
                pos_sim = torch.mm(wo_n, wo_n.T)
                mask = 1.0 - torch.eye(nA, device=DEVICE)
                div_loss = (pos_sim * mask).sum() / (nA * (nA - 1))
            else:
                div_loss = torch.tensor(0.0, device=DEVICE)

            # 3. Coverage: B句每个词都要覆盖
            best_per_B = cos_all[:, :nB_orig].max(dim=0).values
            cov_loss = (1.0 - best_per_B).mean()

            # 4. Contrastive push: 推离随机负样本
            # 选5-10个不在B句中的随机词作为负样本
            neg_cands = [w for w in vocab_pool if w not in B_set]
            n_neg = min(10, len(neg_cands))
            if n_neg > 0:
                neg_picked = random.sample(neg_cands, n_neg)
                neg_vecs = torch.stack([vocab_vecs[vocab_pool.index(w)] for w in neg_picked])
                neg_sim = torch.mm(wo_n, F.normalize(neg_vecs, dim=-1).T)  # [nA, n_neg]
                # Margin: cos应该低于0.2, 超过则惩罚
                contrast_loss = F.relu(neg_sim - 0.2).mean()
            else:
                contrast_loss = torch.tensor(0.0, device=DEVICE)

            # 5. Sentence
            sent_pred_n = F.normalize(word_out.mean(dim=0), dim=-1)
            B_mean_n = F.normalize(B_vecs.mean(dim=0), dim=-1)
            sent_loss = 1.0 - (sent_pred_n * B_mean_n).sum()

            loss = (w_group * group_loss + w_div * div_loss +
                    w_cov * cov_loss + w_contrast * contrast_loss +
                    w_sent * sent_loss)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            em["group"] += group_loss.item(); em["div"] += div_loss.item()
            em["cov"] += cov_loss.item(); em["contrast"] += contrast_loss.item()
            em["sent"] += sent_loss.item(); nb += 1
            last_loss = loss.item()

        for k in em: em[k] /= nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P7-2048 v5] Epoch {epoch:4d}/{ep} | group={em['group']:.4f} "
                  f"div={em['div']:.4f} cov={em['cov']:.4f} "
                  f"ctr={em['contrast']:.4f} sent={em['sent']:.4f} | {elapsed:.1f}s")
            print(f"{'='*70}")

            # 可视化
            A_v, B_v, A_w, B_w = encoded_pairs[0]
            wo, _, attn_map = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1)
            Bv_n = F.normalize(B_v, dim=-1)
            cos_m = torch.mm(wo_n, Bv_n.T)

            print(f"\n[映射] A='{A_w}' -> B='{B_w}'")
            for i, aw in enumerate(A_w):
                row = f"  {aw:6s}:"
                for j in range(len(B_w)):
                    row += f" {B_w[j]}={cos_m[i,j].item():.2f}"
                print(row)
            pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
            print(f"  pred: {pred}")

            # 负样本距离检测
            rand_words = random.sample([w for w in vocab_pool if w not in set(B_w)], 5)
            rand_vecs = torch.stack([vocab_vecs[vocab_pool.index(w)] for w in rand_words])
            rand_sim = torch.mm(wo_n, F.normalize(rand_vecs, dim=-1).T)
            print(f"  cos to random: {[f'{w}={rand_sim.mean(dim=0)[i].item():.2f}' for i,w in enumerate(rand_words)]}")

            print(f"[GPU] {torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | group={em['group']:.4f} div={em['div']:.4f} "
                  f"ctr={em['contrast']:.4f}")

    # ── 保存 ──
    save_path = os.path.join(SAVE_DIR, "P7_2048word_v5.pt")
    torch.save({"model_state_dict": model.state_dict(), "word_dim": 128,
                "attn_dim": 2048, "heads": 2048}, save_path)

    # ── 验收 ──
    print(f"\n{'#'*60}")
    print("# P7-2048 v5 验收")

    # 全候选表检索
    B_cand_words = sorted(set(w for _, _, _, B_w in encoded_pairs for w in B_w))
    B_cand_vecs = [enc(w) for w in B_cand_words if enc(w) is not None]
    B_cand_words = [w for w, v in zip(B_cand_words, B_cand_vecs) if v is not None]
    B_cand_table = torch.stack(B_cand_vecs)

    print(f"\n[全候选表检索] {len(B_cand_words)}词")
    test_inputs = [
        (["我", "喜欢", "苹果"], ["你", "喜欢", "什么"]),
        (["我", "喜欢", "苹果"], ["谁", "喜欢", "苹果"]),
        (["你", "喜欢", "电影"], ["我", "喜欢", "什么"]),
        (["他", "吃", "米饭"], ["他", "吃", "什么"]),
        (["老师", "教", "学生"], ["老师", "教", "什么"]),
    ]
    outputs_set = set()
    for A_w, B_w in test_inputs:
        A_vecs = torch.stack([enc(w) for w in A_w])
        topk_idx, topk_scores, _ = model.retrieve_from_table(A_vecs, B_cand_table, topk=3)
        pred = tuple(B_cand_words[topk_idx[i, 0].item()] for i in range(len(A_w)))
        outputs_set.add(pred)
        print(f"  {A_w} -> {list(pred)} (true={B_w})")

    n_unique_outputs = len(outputs_set)
    print(f"\n  不同输出数: {n_unique_outputs}/{len(test_inputs)} "
          f"{'PASS - 不再万能答案!' if n_unique_outputs > 1 else 'STILL UNIVERSAL'}")

    print(f"{'#'*60}")
    print(f"[保存] {save_path}")


if __name__ == "__main__":
    train()
