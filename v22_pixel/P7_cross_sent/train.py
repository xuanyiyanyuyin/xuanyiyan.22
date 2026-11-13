"""
P7 跨句词级路由 训练
=====================
A词序列 → 交叉注意力(Q=A词,KV=B词表) → B句向量
Loss: cos(pred_B, true_B)
"""
import torch, torch.nn.functional as F, time, os, sys, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P5_sentence.model import SentenceSynthesis
from P7_cross_sent.model import CrossSentenceRouter


def load_pairs(path):
    """加载A-B句对"""
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
    if DEVICE.type != "cuda": return (0,0,0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2,
            torch.cuda.memory_reserved(DEVICE)/1024**2,
            torch.cuda.max_memory_allocated(DEVICE)/1024**2)


def train():
    ep = 300; disp = 50; lr = 0.003
    print(f"P7 跨句路由 epochs={ep} display={disp} lr={lr}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]

    # P5
    p5_path = os.path.join(SAVE_DIR, "P5_final.pt")
    p5 = SentenceSynthesis().to(DEVICE)
    if os.path.exists(p5_path):
        p5_ckpt = torch.load(p5_path, map_location=DEVICE)
        p5.load_state_dict(p5_ckpt.get("model_state_dict", p5_ckpt.get("model", {})))
    for p in p5.parameters(): p.requires_grad = False; p5.eval()

    # 数据
    pairs = load_pairs("C:/ai/P7_cross_sent/data_p7.txt")
    print(f"[数据] {len(pairs)}对A-B句")

    # 编码词
    @torch.no_grad()
    def enc(w):
        c1, c2 = w[0], w[0] if len(w) == 1 else w[1]
        if c1 not in p1_c2i or c2 not in p1_c2i: return None
        return p1(torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    # 收集B词表 (所有出现过的B词)
    B_vocab = set()
    for _, B_words in pairs:
        B_vocab.update(B_words)
    B_vocab_list = sorted(B_vocab)
    B_w2id = {w: i for i, w in enumerate(B_vocab_list)}
    B_table = torch.stack([enc(w) for w in B_vocab_list if enc(w) is not None])  # [vocab_B, 32]
    print(f"[B词表] {len(B_table)}词")

    # 编码所有句对
    encoded_pairs = []
    for A_words, B_words in pairs:
        A_vecs = [enc(w) for w in A_words]
        B_vecs = [enc(w) for w in B_words]
        if None not in A_vecs and None not in B_vecs and len(A_vecs) == 3 and len(B_vecs) == 3:
            encoded_pairs.append((torch.stack(A_vecs), torch.stack(B_vecs), A_words, B_words))
    print(f"[编码] {len(encoded_pairs)}对有效")

    # P7
    model = CrossSentenceRouter().to(DEVICE)
    print(f"[P7] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; best_cos = 0.0; total_t0 = time.time()

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; epoch_cos = 0.0; nb = 0

        for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
            # P7: A词 → 交叉注意力 → B句向量
            B_pred, attn_map = model(A_vecs, B_table, last_loss=last_loss)

            # 真实B: P5合成B句的句子向量
            with torch.no_grad():
                B_true = p5(B_vecs, torch.zeros(3, dtype=torch.long, device=DEVICE), last_loss=0.0)

            loss = 1.0 - F.cosine_similarity(B_pred.unsqueeze(0), B_true.unsqueeze(0), dim=-1)
            cos = 1.0 - loss.item()

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); epoch_cos += cos; nb += 1
            last_loss = loss.item()

        al = epoch_loss / nb; ac = epoch_cos / nb
        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"[P7] Epoch {epoch:4d} / {ep} | Loss: {al:.6f} | cos: {ac:.4f} | 累计: {time.time()-total_t0:.0f}s")
            print(f"{'='*70}")

            # 调制信息
            exp = model.explore_state
            meta = model.meta_fc(exp * min(last_loss*20.0, 1.0))
            print(f"[调制] L2={meta.norm().item():.4f}")

            # 交叉注意力可视化 (取第一对)
            A_vecs, B_vecs, A_words, B_words = encoded_pairs[0]
            _, attn_map = model(A_vecs, B_table, last_loss=0.0)
            attn_avg = attn_map.mean(dim=0)  # [nA, vocab_B]

            print(f"\n[交叉注意力] A='{A_words}' → B='{B_words}'")
            print(f"  {'':12s}", end="")
            for w in B_vocab_list[:15]:
                print(f"{w:6s}", end="")
            print()
            for i, aw in enumerate(A_words):
                print(f"  {aw:12s}", end="")
                for j in range(min(15, len(B_vocab_list))):
                    score = attn_avg[i, j].item()
                    marker = "██" if score > 0.1 else ("▓▓" if score > 0.05 else "··")
                    print(f"{marker:6s}", end="")
                print()
            print(f"  (值越大=该A词对该B词的引力越强)")

            # 采样几个句对看预测
            print(f"\n[预测采样]")
            for A_vecs, B_vecs, A_words, B_words in encoded_pairs[:2]:
                B_pred, _ = model(A_vecs, B_table, last_loss=0.0)
                with torch.no_grad():
                    B_true = p5(B_vecs, torch.zeros(3, dtype=torch.long, device=DEVICE), last_loss=0.0)
                cs = F.cosine_similarity(B_pred.unsqueeze(0), B_true.unsqueeze(0), dim=-1).item()
                print(f"  A='{A_words}' → 预测B cos={cs:.4f} (真实B='{B_words}')")

            a,r,m = gpu_mem()
            print(f"\n[GPU] alloc={a:.0f}MB | {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 25 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f} | cos: {ac:.4f}")

    # 最终测试
    print(f"\n{'#'*60}")
    print("# P7 最终验收")
    cos_list = []
    for A_vecs, B_vecs, A_words, B_words in encoded_pairs:
        B_pred, _ = model(A_vecs, B_table, last_loss=0.0)
        with torch.no_grad():
            B_true = p5(B_vecs, torch.zeros(3, dtype=torch.long, device=DEVICE), last_loss=0.0)
        cs = F.cosine_similarity(B_pred.unsqueeze(0), B_true.unsqueeze(0), dim=-1).item()
        cos_list.append(cs)
    avg = sum(cos_list)/len(cos_list)
    print(f"  平均cos: {avg:.4f}")
    print(f"  验收: cos>0.9 → {'PASS' if avg>0.9 else 'FAIL'}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    train()
