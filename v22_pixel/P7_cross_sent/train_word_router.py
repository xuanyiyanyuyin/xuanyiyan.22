"""
P7 2048头词级路由 训练
========================
Q=A句词 → 2048头交叉注意力(K,V=B句词) → 预测B词向量 + B句向量
三损失: 句级cos + 词级匹配 + 注意力熵正则
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
    ep = 500; disp = 25; lr = 0.002
    print(f"P7 2048头词级路由 epochs={ep} display={disp} lr={lr}\n")

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

    # 收集所有B词
    all_B_words = set()
    for _, B_words in pairs:
        all_B_words.update(B_words)
    B_vocab_list = sorted(all_B_words)
    B_vocab_vecs = [enc(w) for w in B_vocab_list if enc(w) is not None]
    B_vocab_list = [w for w, v in zip(B_vocab_list, B_vocab_vecs) if v is not None]
    B_vocab_table = torch.stack(B_vocab_vecs)
    print(f"[B词表] {len(B_vocab_list)}词")

    # 编码句对
    encoded_pairs = []
    for A_words, B_words in pairs:
        A_vecs = [enc(w) for w in A_words]
        B_vecs = [enc(w) for w in B_words]
        if None not in A_vecs and None not in B_vecs and len(A_vecs) >= 2 and len(B_vecs) >= 2:
            encoded_pairs.append((torch.stack(A_vecs), torch.stack(B_vecs),
                                  A_words, B_words))
    print(f"[编码] {len(encoded_pairs)}对有效\n")

    # ── P7 2048头模型 ──
    model = P7WordRouter2048().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[P7-2048] 参数: {n_params:,}  (heads=2048, head_dim=1)")

    # 初始化: Q/K/V投影加对角分量 → 初始≈余弦
    for proj in [model.q_proj, model.k_proj, model.v_proj]:
        nn.init.normal_(proj.weight, 0, 0.02)
        with torch.no_grad():
            proj.weight.data += torch.eye(2048, 128, device=DEVICE) * 0.1
    nn.init.normal_(model.out_proj.weight, 0, 0.02)
    nn.init.normal_(model.sent_proj.weight, 0, 0.02)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    last_loss = 1.0; total_t0 = time.time()

    print(f"{'='*70}")
    print(f"  训练: {len(encoded_pairs)}对 × {ep}轮")
    print(f"  Loss = sent_cos + 0.3×word_cos + 0.05×ent_penalty")
    print(f"{'='*70}\n")

    for epoch in range(1, ep + 1):
        t0 = time.time()
        epoch_sent_loss = 0.0; epoch_word_loss = 0.0
        epoch_ent_loss = 0.0; epoch_cos = 0.0; nb = 0

        for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
            nA, nB = A_vecs.shape[0], B_vecs.shape[0]

            # 前向: word_out [nA,128], sent_vec [256], attn [2048,nA,nB]
            word_out, sent_vec, attn = model(A_vecs, B_vecs, last_loss=last_loss)

            # Loss 1: 句级 cos — sent_vec vs P5 B句向量
            with torch.no_grad():
                sent_true = p5(B_vecs, torch.zeros(nB, dtype=torch.long, device=DEVICE),
                              last_loss=0.0)
            sent_loss = 1.0 - F.cosine_similarity(
                sent_vec.unsqueeze(0), sent_true.unsqueeze(0), dim=-1)

            # Loss 2: 词级 — 每A输出与最近B词匹配
            word_out_n = F.normalize(word_out, dim=-1)
            B_vecs_n = F.normalize(B_vecs, dim=-1)
            word_sim = torch.mm(word_out_n, B_vecs_n.T)  # [nA, nB]
            best_match = word_sim.max(dim=-1).values  # [nA]
            word_loss = (1.0 - best_match).mean()

            # Loss 3: 注意力熵 (鼓励聚焦)
            attn_avg = attn.mean(dim=0)  # [nA, nB]
            ent = -(attn_avg * torch.log(attn_avg + 1e-8)).sum(dim=-1).mean()
            ent_loss = F.mse_loss(ent, torch.tensor(0.5, device=DEVICE))

            loss = sent_loss + 0.3 * word_loss + 0.05 * ent_loss
            cos = 1.0 - sent_loss.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_sent_loss += sent_loss.item()
            epoch_word_loss += word_loss.item()
            epoch_ent_loss += ent.item()
            epoch_cos += cos; nb += 1
            last_loss = loss.item()

        al = epoch_sent_loss / nb; wl = epoch_word_loss / nb
        el = epoch_ent_loss / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P7-2048] Epoch {epoch:4d}/{ep} | sent={al:.4f} word={wl:.4f} "
                  f"ent={el:.4f} | cos={ac:.4f} | {elapsed:.1f}s")
            print(f"{'='*70}")

            print(f"[调制] explore={model.explore_state.norm().item():.3f} "
                  f"temp={model.temperature.item():.3f}")

            # 注意力可视化
            A_v, B_v, A_w, B_w = encoded_pairs[0]
            _, _, attn_map = model(A_v, B_v, last_loss=0.0)
            attn_avg = attn_map.mean(dim=0)

            print(f"\n[词级注意力] A='{A_w}' → B='{B_w}'")
            header = "  " + "".join(f"{w:>10s}" for w in B_w)
            print(header)
            for i, aw in enumerate(A_w):
                row = f"  {aw:6s} "
                for j in range(len(B_w)):
                    s = attn_avg[i, j].item()
                    bar = "█" * int(s * 40) if s > 0.02 else "·"
                    row += f"{s:.3f}{bar:8s}"
                print(row)

            # 词级预测
            print(f"\n[词级预测] A位置→最近B词:")
            wo, _, _ = model(A_v, B_v, last_loss=0.0)
            wo_n = F.normalize(wo, dim=-1); Bv_n = F.normalize(B_v, dim=-1)
            sim = torch.mm(wo_n, Bv_n.T)
            for i, aw in enumerate(A_w):
                top_j = sim[i].argmax().item()
                print(f"  {aw} → {B_w[top_j]} (cos={sim[i,top_j].item():.3f})")

            # 采样
            print(f"\n[采样预测]")
            for A_v, B_v, A_w, B_w in encoded_pairs[:3]:
                wo, _, _ = model(A_v, B_v, last_loss=0.0)
                wo_n = F.normalize(wo, dim=-1); Bv_n = F.normalize(B_v, dim=-1)
                s = torch.mm(wo_n, Bv_n.T)
                pred = [B_w[s[i].argmax().item()] for i in range(len(A_w))]
                print(f"  {A_w} → {pred}  (true={B_w})")

            a, r, m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB | reserved={r:.0f}MB")
            print(f"{'='*70}\n")
        elif epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | sent={al:.4f} word={wl:.4f} | cos={ac:.4f}")

    # ── 最终验收 ──
    print(f"\n{'#'*60}")
    print("# P7-2048 最终验收")
    cos_list = []
    for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
        _, sent_vec, _ = model(A_vecs, B_vecs, last_loss=0.0)
        with torch.no_grad():
            sent_true = p5(B_vecs, torch.zeros(len(B_words), dtype=torch.long, device=DEVICE),
                          last_loss=0.0)
        cs = F.cosine_similarity(sent_vec.unsqueeze(0), sent_true.unsqueeze(0), dim=-1).item()
        cos_list.append(cs)

    avg_cos = sum(cos_list) / len(cos_list)
    print(f"  句级cos: {avg_cos:.4f}  {'PASS' if avg_cos > 0.85 else 'TRAIN MORE'}")
    print(f"{'#'*60}")

    # 保存
    save_path = os.path.join(SAVE_DIR, "P7_2048word_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_dim": 128, "attn_dim": 2048, "heads": 2048,
        "B_vocab_list": B_vocab_list,
        "avg_cos": avg_cos
    }, save_path)
    print(f"[保存] {save_path}")


if __name__ == "__main__":
    train()
