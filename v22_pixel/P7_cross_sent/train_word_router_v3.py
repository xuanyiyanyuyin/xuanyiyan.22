"""
P7 2048头词级路由 训练 v3 — 词组级监督 (非硬对齐)
==================================================
Group set loss: 每个pred匹配B句中某词 → 词组对词组
Diversity loss: 不同位置输出必须不同 → 防坍缩
Coverage loss: B句每个词都要被覆盖 → 完整性
Sentence loss: 低权重辅助
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os, sys

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
    ep = 300; disp = 20; lr = 0.001
    print(f"P7-2048 v3 词组监督 epochs={ep} lr={lr}\n")

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
    print(f"[编码] {len(encoded_pairs)}对有效\n")

    # ── 模型 ──
    model = P7WordRouter2048().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7-2048 v3] 参数: {n_params:,}")

    nn.init.normal_(model.q_proj.weight, 0, 0.02)
    for proj in [model.k_proj, model.v_proj]:
        nn.init.normal_(proj.weight, 0, 0.02)
        with torch.no_grad():
            proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
    nn.init.normal_(model.out_proj.weight, 0, 0.02)
    nn.init.normal_(model.pos_transform.weight, 0, 0.02)

    model.temperature.data.fill_(2.0)
    model.temperature.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    last_loss = 1.0; total_t0 = time.time()

    # 权重: group > diversity ≈ coverage > sentence
    w_group = 0.5; w_div = 0.3; w_cov = 0.2; w_sent = 0.1; w_pos = 0.05

    print(f"{'='*70}")
    print(f"  Loss = {w_group}×group + {w_div}×diversity + {w_cov}×coverage")
    print(f"       + {w_sent}×sentence + {w_pos}×position_aux")
    print(f"  K/V = B句具体词 (词组级映射)")
    print(f"{'='*70}\n")

    for epoch in range(1, ep + 1):
        t0 = time.time()
        em = {"group": 0, "div": 0, "cov": 0, "sent": 0, "pos": 0}
        nb = 0

        for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
            nA, nB = A_vecs.shape[0], B_vecs.shape[0]

            # 前向: K/V = B句具体词 (词组级)
            word_out, sent_vec, attn = model(A_vecs, B_vecs, last_loss=last_loss)
            # word_out: [nA, 128], attn: [2048, nA, nB]

            wo_n = F.normalize(word_out, dim=-1)
            Bv_n = F.normalize(B_vecs, dim=-1)
            cos_all = torch.mm(wo_n, Bv_n.T)  # [nA, nB]

            # 1. Group set loss: 每个A输出匹配B句中某词
            best_per_A = cos_all.max(dim=-1).values  # [nA]
            group_loss = (1.0 - best_per_A).mean()

            # 2. Diversity loss: 不同A位置输出必须不同
            # 计算输出向量间的成对余弦相似度
            if nA > 1:
                pos_sim = torch.mm(wo_n, wo_n.T)  # [nA, nA]
                mask = 1.0 - torch.eye(nA, device=DEVICE)
                div_loss = (pos_sim * mask).sum() / (nA * (nA - 1))
            else:
                div_loss = torch.tensor(0.0, device=DEVICE)

            # 3. Coverage loss: B句每个词都要被覆盖到
            best_per_B = cos_all.max(dim=0).values  # [nB]
            cov_loss = (1.0 - best_per_B).mean()

            # 4. Sentence loss: 低权重辅助
            sent_pred_n = F.normalize(word_out.mean(dim=0), dim=-1)
            B_mean_n = F.normalize(B_vecs.mean(dim=0), dim=-1)
            sent_loss = 1.0 - (sent_pred_n * B_mean_n).sum()

            # 5. Position auxiliary loss: 仅同长度句子的同位辅助
            pos_loss = torch.tensor(0.0, device=DEVICE)
            if nA == nB:
                diag_cos = cos_all.diag().mean()
                pos_loss = 1.0 - diag_cos

            loss = (w_group * group_loss + w_div * div_loss +
                    w_cov * cov_loss + w_sent * sent_loss +
                    w_pos * pos_loss)

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
            print(f"[P7-2048 v3] Epoch {epoch:4d}/{ep} | group={em['group']:.4f} "
                  f"div={em['div']:.4f} cov={em['cov']:.4f} sent={em['sent']:.4f} "
                  f"pos={em['pos']:.4f} | {elapsed:.1f}s")
            print(f"{'='*70}")

            # 可视化第一对
            A_v, B_v, A_w, B_w = encoded_pairs[0]
            wo, _, attn_map = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1)
            Bv_n = F.normalize(B_v, dim=-1)
            cos_m = torch.mm(wo_n, Bv_n.T)
            attn_avg = attn_map.mean(dim=0)

            print(f"\n[词组映射] A='{A_w}' → B='{B_w}'")
            print(f"  {'':8s}", end="")
            for bw in B_w: print(f"{bw:>10s}", end="")
            print("\n  cosine:")
            for i, aw in enumerate(A_w):
                print(f"  {aw:6s} ", end="")
                for j in range(len(B_w)):
                    print(f"{cos_m[i,j].item():>10.3f}", end="")
                print()
            print("  attention:")
            for i, aw in enumerate(A_w):
                print(f"  {aw:6s} ", end="")
                for j in range(len(B_w)):
                    print(f"{attn_avg[i,j].item():>10.3f}", end="")
                print()

            # 每个A位置的最佳匹配B词
            best_j = cos_m.argmax(dim=-1)
            pred_words = [B_w[j.item()] for j in best_j]
            print(f"  pred: {A_w} → {pred_words}  (true={B_w})")

            # 多样性检查
            if len(A_w) > 1:
                pair_sims = []
                for i1 in range(len(A_w)):
                    for i2 in range(i1+1, len(A_w)):
                        pair_sims.append((wo_n[i1] * wo_n[i2]).sum().item())
                print(f"  pairwise cos between outputs: {[f'{s:.3f}' for s in pair_sims]}")

            # 采样
            print(f"\n[采样]")
            for A_v, B_v, A_w, B_w in encoded_pairs[1:4]:
                wo, _, _ = model(A_v, B_v, last_loss=0.0)
                wo_n = F.normalize(wo, dim=-1)
                Bv_n = F.normalize(B_v, dim=-1)
                cos_m = torch.mm(wo_n, Bv_n.T)
                pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
                print(f"  {A_w} → {pred}  (true={B_w})")

            print(f"[GPU] {torch.cuda.memory_allocated(DEVICE)/1024**2:.0f}MB")
            print(f"{'='*70}\n")
        elif epoch % 40 == 0:
            print(f"  Epoch {epoch:4d} | group={em['group']:.4f} div={em['div']:.4f} "
                  f"cov={em['cov']:.4f} sent={em['sent']:.4f}")

    # ── 验收 ──
    print(f"\n{'#'*60}")
    print("# P7-2048 v3 验收")
    total_uniq = 0; total_pairs = 0
    for A_v, B_v, A_w, B_w in encoded_pairs:
        wo, _, _ = model(A_v, B_v, last_loss=0.0)
        wo_n = F.normalize(wo, dim=-1)
        Bv_n = F.normalize(B_v, dim=-1)
        cos_m = torch.mm(wo_n, Bv_n.T)
        pred = [B_w[cos_m[i].argmax().item()] for i in range(len(A_w))]
        uniq = len(set(pred))
        total_uniq += uniq; total_pairs += 1
    avg_uniq = total_uniq / total_pairs
    print(f"  平均独特预测词数: {avg_uniq:.2f} (目标>1.5, 表示没有全坍缩)")
    print(f"  坍缩检查: {'PASS' if avg_uniq > 1.3 else 'STILL COLLAPSED'}")
    print(f"{'#'*60}")

    save_path = os.path.join(SAVE_DIR, "P7_2048word_v3.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_dim": 128, "attn_dim": 2048, "heads": 2048,
        "avg_unique": avg_uniq
    }, save_path)
    print(f"[保存] {save_path}")


if __name__ == "__main__":
    train()
