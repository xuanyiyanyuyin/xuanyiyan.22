"""
P7 2048头词级路由 训练 v2
===========================
修复: K/V=全B词表(84词)而非仅B句3词 → 2048头有足够多样性
新增: 位置多样性loss → 不同A位置关注不同B词
Q=A句词 → 2048头交叉注意力(K,V=全词表) → 检索B词
Loss = sent_cos + word_retrieval + position_diversity
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
            A_words = parts[0].split()
            B_words = parts[1].split()
            pairs.append((A_words, B_words))
    return pairs


def gpu_mem():
    if DEVICE.type != "cuda": return (0, 0, 0)
    return (torch.cuda.memory_allocated(DEVICE) / 1024**2,
            torch.cuda.memory_reserved(DEVICE) / 1024**2,
            torch.cuda.max_memory_allocated(DEVICE) / 1024**2)


def train():
    ep = 400; disp = 25; lr = 0.001
    print(f"P7 2048头词级路由 v2 epochs={ep} display={disp} lr={lr}\n")

    # ── P1 (frozen) ──
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # ── P5 (frozen) ──
    p5_path = os.path.join(SAVE_DIR, "P5_final.pt")
    p5 = SentenceSynthesis().to(DEVICE)
    if os.path.exists(p5_path):
        p5_ckpt = torch.load(p5_path, map_location=DEVICE)
        p5.load_state_dict(p5_ckpt.get("model_state_dict", p5_ckpt.get("model", {})))
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

    # ── 数据 ──
    pairs = load_pairs("C:/ai/P7_cross_sent/data_p7_v2.txt")
    print(f"[数据] {len(pairs)}对A-B句")

    @torch.no_grad()
    def enc(w):
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    # 全B词表
    all_B_words = set()
    for _, B_words in pairs:
        all_B_words.update(B_words)
    B_vocab_list = sorted(all_B_words)
    B_vocab_vecs = [enc(w) for w in B_vocab_list if enc(w) is not None]
    B_vocab_list = [w for w, v in zip(B_vocab_list, B_vocab_vecs) if v is not None]
    B_vocab_table = torch.stack(B_vocab_vecs)  # [vocab_B, 128]
    B_w2id = {w: i for i, w in enumerate(B_vocab_list)}
    print(f"[B词表] {len(B_vocab_list)}词")

    # 编码句对
    encoded_pairs = []
    for A_words, B_words in pairs:
        A_vecs = [enc(w) for w in A_words]
        B_vecs = [enc(w) for w in B_words]
        if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
            # B词在词表中的索引 → 用于检索loss
            B_ids = torch.tensor([B_w2id[w] for w in B_words if w in B_w2id], device=DEVICE)
            encoded_pairs.append((torch.stack(A_vecs), torch.stack(B_vecs),
                                  A_words, B_words, B_ids))
    print(f"[编码] {len(encoded_pairs)}对有效\n")

    # ── P7 2048头模型 ──
    model = P7WordRouter2048().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7-2048] 参数: {n_params:,}  (heads=2048, head_dim=1)")

    # 初始化: q_proj (256→2048), k_proj/v_proj (128→2048)
    nn.init.normal_(model.q_proj.weight, 0, 0.02)
    for proj in [model.k_proj, model.v_proj]:
        nn.init.normal_(proj.weight, 0, 0.02)
        with torch.no_grad():
            proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
    nn.init.normal_(model.out_proj.weight, 0, 0.02)
    nn.init.normal_(model.sent_proj.weight, 0, 0.02)
    nn.init.normal_(model.pos_transform.weight, 0, 0.02)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    last_loss = 1.0; total_t0 = time.time()

    print(f"{'='*70}")
    print(f"  训练: {len(encoded_pairs)}对 × {ep}轮 | K/V=全词表({len(B_vocab_list)}词)")
    print(f"  Loss = sent_cos + 0.3×word_retrieval + 0.1×pos_diversity + 0.02×ent")
    print(f"{'='*70}\n")

    for epoch in range(1, ep + 1):
        t0 = time.time()
        epoch_sent = 0.0; epoch_word = 0.0
        epoch_div = 0.0; epoch_ent = 0.0; epoch_cos = 0.0; nb = 0

        for A_vecs, B_vecs, A_words, B_words, B_ids in encoded_pairs:
            nA, nB_true = A_vecs.shape[0], B_vecs.shape[0]

            # 前向: K=V=全B词表 (非仅B句的3词!)
            word_out, sent_vec, attn = model(A_vecs, B_vocab_table, last_loss=last_loss)
            # word_out: [nA, 128], sent_vec: [256], attn: [2048, nA, vocab_B]

            # Loss 1: 句级 — sent_vec vs P5 B句向量
            with torch.no_grad():
                sent_true = p5(B_vecs, torch.zeros(nB_true, dtype=torch.long, device=DEVICE),
                              last_loss=0.0)
            sent_loss = 1.0 - F.cosine_similarity(
                sent_vec.unsqueeze(0), sent_true.unsqueeze(0), dim=-1)

            # Loss 2: 词检索 — 每A位置输出应匹配正确的B词
            word_out_n = F.normalize(word_out, dim=-1)
            B_tab_n = F.normalize(B_vocab_table, dim=-1)
            cos_all = torch.mm(word_out_n, B_tab_n.T)  # [nA, vocab_B]

            # 对每个A位置，取它到所有正确B词的最大cos，平均后1-cos
            retrieval_scores = []
            for i in range(nA):
                correct_cos = cos_all[i, B_ids]  # B_ids: 正确B词在词表中的索引
                retrieval_scores.append(correct_cos.max())
            word_loss = 1.0 - torch.stack(retrieval_scores).mean()

            # Loss 3: 位置多样性 — 不同A位置应关注不同B词
            attn_avg = attn.mean(dim=0)  # [nA, vocab_B]
            # 计算位置间的注意力余弦相似度，希望它们不同
            attn_flat = attn_avg.view(nA, -1)  # [nA, vocab_B]
            pos_sim = torch.mm(attn_flat, attn_flat.T)  # [nA, nA]
            mask = 1.0 - torch.eye(nA, device=DEVICE)
            diversity_loss = (pos_sim * mask).sum() / (nA * (nA - 1) + 1e-8)

            # Loss 4: 熵 (温和正则)
            ent = -(attn_avg * torch.log(attn_avg + 1e-8)).sum(dim=-1).mean()
            ent_loss = F.mse_loss(ent, torch.tensor(1.5, device=DEVICE))

            loss = sent_loss + 0.3 * word_loss + 0.1 * diversity_loss + 0.02 * ent_loss
            cos = 1.0 - sent_loss.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_sent += sent_loss.item(); epoch_word += word_loss.item()
            epoch_div += diversity_loss.item(); epoch_ent += ent.item()
            epoch_cos += cos; nb += 1
            last_loss = loss.item()

        al = epoch_sent / nb; wl = epoch_word / nb
        dl = epoch_div / nb; el = epoch_ent / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P7-2048 v2] Epoch {epoch:4d}/{ep} | sent={al:.4f} word={wl:.4f} "
                  f"div={dl:.4f} ent={el:.4f} | cos={ac:.4f} | {elapsed:.1f}s")
            print(f"{'='*70}")

            # 注意力 & 检索可视化
            A_v, B_v, A_w, B_w, B_id = encoded_pairs[0]
            wo, _, attn_map = model(A_v, B_vocab_table, last_loss=0.0)
            attn_avg = attn_map.mean(dim=0)  # [nA, vocab_B]

            # 显示每A位置关注的前5个B词
            print(f"\n[注意力] A='{A_w}' → true_B='{B_w}'")
            for i, aw in enumerate(A_w):
                top5 = torch.topk(attn_avg[i], 5)
                tops = [(B_vocab_list[idx.item()], f"{attn_avg[i,idx].item():.3f}")
                        for idx in top5.indices]
                print(f"  {aw:6s} → {tops}")

            # 检索结果
            wo_n = F.normalize(wo, dim=-1)
            cos_all = torch.mm(wo_n, B_tab_n.T)
            print(f"\n[检索] 每位置输出→最近B词:")
            for i, aw in enumerate(A_w):
                top5 = torch.topk(cos_all[i], 5)
                tops = [(B_vocab_list[idx.item()], f"{cos_all[i,idx].item():.3f}")
                        for idx in top5.indices]
                print(f"  {aw:6s} → {tops}")

            # 采样
            print(f"\n[采样]")
            for A_v, B_v, A_w, B_w, B_id in encoded_pairs[:3]:
                wo, _, _ = model(A_v, B_vocab_table, last_loss=0.0)
                wo_n = F.normalize(wo, dim=-1)
                s = torch.mm(wo_n, B_tab_n.T)
                pred = [B_vocab_list[s[i].argmax().item()] for i in range(len(A_w))]
                print(f"  {A_w} → {pred}  (true={B_w})")

            a, r, m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB | {elapsed:.1f}s/epoch")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | sent={al:.4f} word={wl:.4f} div={dl:.4f} | cos={ac:.4f}")

    # ── 验收 ──
    print(f"\n{'#'*60}")
    print("# P7-2048 v2 最终验收")
    cos_list = []
    for A_vecs, B_vecs, A_words, B_words, B_ids in encoded_pairs:
        _, sent_vec, _ = model(A_vecs, B_vocab_table, last_loss=0.0)
        with torch.no_grad():
            sent_true = p5(B_vecs, torch.zeros(len(B_words), dtype=torch.long, device=DEVICE),
                          last_loss=0.0)
        cs = F.cosine_similarity(sent_vec.unsqueeze(0), sent_true.unsqueeze(0), dim=-1).item()
        cos_list.append(cs)
    avg_cos = sum(cos_list) / len(cos_list)
    print(f"  句级cos: {avg_cos:.4f}  {'PASS' if avg_cos > 0.85 else 'TRAIN MORE'}")
    print(f"{'#'*60}")

    save_path = os.path.join(SAVE_DIR, "P7_2048word_v2.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_dim": 128, "attn_dim": 2048, "heads": 2048,
        "B_vocab_list": B_vocab_list,
        "avg_cos": avg_cos
    }, save_path)
    print(f"[保存] {save_path}")


if __name__ == "__main__":
    train()
