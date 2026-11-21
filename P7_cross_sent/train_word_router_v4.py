"""
P7 2048头词级路由 训练 v4 — 反万能答案
========================================
v3问题: 训练数据统一(全3词句) → P7学会位置记忆而非内容路由
v4修复: 随机插入干扰词到K/V → 强迫内容判别
        + 两阶段训练: 先易后难
        + 变长句增强
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
    ep = 400; disp = 20; lr = 0.001
    print(f"P7-2048 v4 反万能答案 epochs={ep} lr={lr}\n")

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

    # ── P5 ──
    p5_path = os.path.join(SAVE_DIR, "P5_final.pt")
    p5 = SentenceSynthesis().to(DEVICE)
    if os.path.exists(p5_path):
        p5_ckpt = torch.load(p5_path, map_location=DEVICE)
        p5.load_state_dict(p5_ckpt.get("model_state_dict", p5_ckpt.get("model", {})))
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

    # ── 数据 ──
    pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_v2.txt")
    print(f"[数据] {len(pairs)}对")

    encoded_pairs = []
    for A_words, B_words in pairs:
        A_vecs = [enc(w) for w in A_words]
        B_vecs = [enc(w) for w in B_words]
        if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
            encoded_pairs.append((torch.stack(A_vecs), torch.stack(B_vecs),
                                  A_words, B_words))
    print(f"[编码] {len(encoded_pairs)}对有效")

    # 收集全词表用作干扰词池
    all_words = set()
    for _, B_words in pairs:
        all_words.update(B_words)
    for A_words, _ in pairs:
        all_words.update(A_words)
    vocab_pool = sorted(all_words)
    vocab_vecs = [enc(w) for w in vocab_pool if enc(w) is not None]
    vocab_pool = [w for w, v in zip(vocab_pool, vocab_vecs) if v is not None]
    vocab_table = torch.stack(vocab_vecs)
    print(f"[干扰词池] {len(vocab_pool)}词\n")

    # ── 模型 ──
    model = P7WordRouter2048().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7-2048 v4] 参数: {n_params:,}")

    nn.init.normal_(model.q_proj.weight, 0, 0.02)
    for proj in [model.k_proj, model.v_proj]:
        nn.init.normal_(proj.weight, 0, 0.02)
        with torch.no_grad():
            proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
    nn.init.normal_(model.out_proj.weight, 0, 0.02)
    # pos_weight learnable scalar (no init needed)
    model.temperature.data.fill_(2.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    last_loss = 1.0

    # 权重: 强diversity防止位置坍缩
    w_group = 0.5; w_div = 0.4; w_cov = 0.2; w_sent = 0.05; w_pos = 0.02

    print(f"{'='*70}")
    print(f"  训练: {len(encoded_pairs)}对 x {ep}轮")
    print(f"  策略: Epoch 1-80 纯B句词, Epoch 81+ 混入2-4个干扰词")
    print(f"  Loss = {w_group}*group + {w_div}*diversity + {w_cov}*coverage")
    print(f"{'='*70}\n")

    for epoch in range(1, ep + 1):
        t0 = time.time()
        em = {"group": 0, "div": 0, "cov": 0, "sent": 0, "pos": 0}; nb = 0

        # 干扰词数量: epoch>80后逐步增加
        if epoch <= 80:
            n_distractors = 0
        elif epoch <= 160:
            n_distractors = 2
        elif epoch <= 250:
            n_distractors = 3
        else:
            n_distractors = 4

        for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
            nA, nB_orig = A_vecs.shape[0], B_vecs.shape[0]

            # ── 构建带干扰词的K/V ──
            if n_distractors > 0:
                # 随机选干扰词 (排除当前B句中已有的词)
                distractors = []
                B_set = set(B_words)
                candidates = [w for w in vocab_pool if w not in B_set]
                if len(candidates) >= n_distractors:
                    picked = random.sample(candidates, n_distractors)
                    for w in picked:
                        idx = vocab_pool.index(w)
                        distractors.append(vocab_table[idx])
                if distractors:
                    B_with_dist = torch.cat([B_vecs] + [d.unsqueeze(0) for d in distractors], dim=0)
                else:
                    B_with_dist = B_vecs
            else:
                B_with_dist = B_vecs

            nB_total = B_with_dist.shape[0]

            # 前向: K/V = B句词 + 干扰词
            word_out, sent_vec, attn = model(A_vecs, B_with_dist, last_loss=last_loss)
            # word_out: [nA, 128], attn: [2048, nA, nB_total]

            wo_n = F.normalize(word_out, dim=-1)
            Bv_n = F.normalize(B_with_dist, dim=-1)
            cos_all = torch.mm(wo_n, Bv_n.T)  # [nA, nB_total]

            # 1. Group set loss: 仅对原始B词(前nB_orig个)计算
            best_per_A = cos_all[:, :nB_orig].max(dim=-1).values  # [nA]
            group_loss = (1.0 - best_per_A).mean()

            # 2. Diversity loss: 不同A位置输出必须不同
            if nA > 1:
                pos_sim = torch.mm(wo_n, wo_n.T)
                mask = 1.0 - torch.eye(nA, device=DEVICE)
                div_loss = (pos_sim * mask).sum() / (nA * (nA - 1))
            else:
                div_loss = torch.tensor(0.0, device=DEVICE)

            # 3. Coverage loss: B句每个原始词都要覆盖
            best_per_B = cos_all[:, :nB_orig].max(dim=0).values
            cov_loss = (1.0 - best_per_B).mean()

            # 4. Anti-distractor loss: 对干扰词(nB_orig之后)的关注度应该低
            anti_dist_loss = torch.tensor(0.0, device=DEVICE)
            if nB_total > nB_orig:
                distractor_attn = attn.mean(dim=0)[:, nB_orig:]  # [nA, n_dist]
                anti_dist_loss = distractor_attn.sum() / (nA * max(1, nB_total - nB_orig))

            # 5. Sentence loss: 低权重
            sent_pred_n = F.normalize(word_out.mean(dim=0), dim=-1)
            B_mean_n = F.normalize(B_vecs.mean(dim=0), dim=-1)
            sent_loss = 1.0 - (sent_pred_n * B_mean_n).sum()

            # 6. Position auxiliary
            pos_loss = torch.tensor(0.0, device=DEVICE)
            if nA == nB_orig:
                pos_loss = 1.0 - cos_all[:, :nB_orig].diag().mean()

            loss = (w_group * group_loss + w_div * div_loss +
                    w_cov * cov_loss + 0.1 * anti_dist_loss +
                    w_sent * sent_loss + w_pos * pos_loss)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            em["group"] += group_loss.item(); em["div"] += div_loss.item()
            em["cov"] += cov_loss.item(); em["sent"] += sent_loss.item()
            em["pos"] += pos_loss.item(); nb += 1
            last_loss = loss.item()

        for k in em: em[k] /= nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P7-2048 v4] Epoch {epoch:4d}/{ep} (n_dist={n_distractors}) | "
                  f"group={em['group']:.4f} div={em['div']:.4f} cov={em['cov']:.4f} "
                  f"sent={em['sent']:.4f} | {elapsed:.1f}s")
            print(f"{'='*70}")

            # 可视化
            A_v, B_v, A_w, B_w = encoded_pairs[0]
            # 展示时用纯B句词(不加干扰), 看注意力质量
            wo, _, attn_map = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1)
            Bv_n = F.normalize(B_v, dim=-1)
            cos_m = torch.mm(wo_n, Bv_n.T)
            attn_avg = attn_map.mean(dim=0)

            print(f"\n[映射] A='{A_w}' -> B='{B_w}'")
            print(f"  cosine matrix:")
            print(f"  {'':8s}", end="")
            for bw in B_w: print(f"{bw:>10s}", end="")
            print()
            for i, aw in enumerate(A_w):
                print(f"  {aw:6s} ", end="")
                for j in range(len(B_w)):
                    print(f"{cos_m[i,j].item():>10.3f}", end="")
                print()

            best_j = cos_m.argmax(dim=-1)
            pred = [B_w[j.item()] for j in best_j]
            print(f"  pred: {A_w} -> {pred}  (true={B_w})")

            if len(A_w) > 1:
                pairs_sim = []
                for i1 in range(len(A_w)):
                    for i2 in range(i1+1, len(A_w)):
                        pairs_sim.append((wo_n[i1] * wo_n[i2]).sum().item())
                print(f"  pairwise cos: {[f'{s:.3f}' for s in pairs_sim]}")

            # 采样
            print(f"\n[采样]")
            for A_v, B_v, A_w, B_w in encoded_pairs[1:4]:
                wo, _, _ = model(A_v, B_v, last_loss=0.0)
                wo_n = F.normalize(wo, dim=-1)
                Bv_n = F.normalize(B_v, dim=-1)
                cos_m = torch.mm(wo_n, Bv_n.T)
                pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
                print(f"  {A_w} -> {pred}  (true={B_w})")

            print(f"[GPU] {torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB")
            print(f"{'='*70}\n")
        elif epoch % 40 == 0:
            print(f"  Epoch {epoch:4d} (n_dist={n_distractors}) | "
                  f"group={em['group']:.4f} div={em['div']:.4f} cov={em['cov']:.4f}")

    # ── 先保存 ──
    save_path = os.path.join(SAVE_DIR, "P7_2048word_v4.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_dim": 128, "attn_dim": 2048, "heads": 2048,
    }, save_path)

    # ── 验收 ──
    print(f"\n{'#'*60}")
    print("# P7-2048 v4 验收")

    # 测试1: 纯B句词 (训练模式)
    print("\n[测试1: 纯B句词 (训练分布)]")
    for A_v, B_v, A_w, B_w in encoded_pairs[:5]:
        wo, _, _ = model(A_v, B_v, last_loss=0.0)
        wo_n = F.normalize(wo, dim=-1)
        Bv_n = F.normalize(B_v, dim=-1)
        cos_m = torch.mm(wo_n, Bv_n.T)
        pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
        uniq = len(set(pred))
        print(f"  {A_w} -> {pred} (true={B_w}) unique={uniq}")

    # 测试2: B句词+干扰词 (泛化测试)
    print(f"\n[测试2: B句词+3干扰词 (泛化)]")
    total_uniq = 0; total_samples = 0
    for A_v, B_v, A_w, B_w in encoded_pairs:
        # 随机加干扰
        B_set = set(B_w)
        cands = [w for w in vocab_pool if w not in B_set]
        if len(cands) >= 3:
            picked = random.sample(cands, 3)
            dist_vecs = [vocab_table[vocab_pool.index(w)] for w in picked]
            B_test = torch.cat([B_v] + [d.unsqueeze(0) for d in dist_vecs], dim=0)
            B_test_words = B_w + picked
        else:
            B_test = B_v
            B_test_words = B_w

        wo, _, _ = model(A_v, B_test, last_loss=0.0)
        wo_n = F.normalize(wo, dim=-1)
        Bt_n = F.normalize(B_test, dim=-1)
        cos_m = torch.mm(wo_n, Bt_n.T)
        pred = [B_test_words[cos_m[i].argmax().item()] for i in range(len(A_w))]
        uniq = len(set(pred))
        total_uniq += uniq; total_samples += 1

    avg_uniq = total_uniq / total_samples
    print(f"  平均独特预测: {avg_uniq:.2f} (目标>1.5)")

    # 测试3: 全候选表检索 (最终QA场景)
    print(f"\n[测试3: 全候选表检索 (QA场景)]")
    # 构建候选表
    B_cand_words = sorted(set(w for _, _, _, B_w in encoded_pairs for w in B_w))
    B_cand_vecs = [enc(w) for w in B_cand_words if enc(w) is not None]
    B_cand_words = [w for w, v in zip(B_cand_words, B_cand_vecs) if v is not None]
    B_cand_table = torch.stack(B_cand_vecs)

    for A_v, B_v, A_w, B_w in encoded_pairs[:5]:
        topk_idx, topk_scores, _ = model.retrieve_from_table(A_v, B_cand_table, topk=3)
        pred = [B_cand_words[topk_idx[i, 0].item()] for i in range(len(A_w))]
        print(f"  {A_w} -> {pred} (true={B_w})")

    print(f"\n{'#'*60}")

    avg_unique_str = f"{avg_uniq:.4f}" if 'avg_uniq' in dir() else "N/A"
    print(f"[已保存] {save_path} | avg_unique={avg_unique_str}")


if __name__ == "__main__":
    train()
