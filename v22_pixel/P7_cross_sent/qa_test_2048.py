"""
P7-2048 词级路由 QA 集成测试
=============================
加载训练好的P7-2048模型 → 问题词编码 → retrieve_from_table检索答案词
"""
import torch, torch.nn.functional as F, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P7_cross_sent.model import P7WordRouter2048

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "logs", f"p7_2048_qa_test.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")


def w(s):
    log.write(s + "\n"); print(s)


def main():
    w("=== P7-2048 词级路由 QA 测试 ===")

    # ── P1 ──
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    ck = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(ck["num_chars"], ck["num_words"]).to(DEVICE)
    p1.load_state_dict(ck["model_state_dict"])
    p11p = os.path.join(SAVE_DIR, "P11_joint_best.pt")
    if os.path.exists(p11p):
        p1.output_proj.load_state_dict(torch.load(p11p, map_location=DEVICE)["p1_proj"])
        w("[P1] + P11 projection loaded")
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    c2i = ck["char2idx"]; i2w = ck["idx2word"]

    # ── P7-2048 ──
    p7_path = os.path.join(SAVE_DIR, "P7_2048word_best.pt")
    p7_ckpt = torch.load(p7_path, map_location=DEVICE)
    model = P7WordRouter2048().to(DEVICE)
    model.load_state_dict(p7_ckpt["model_state_dict"])
    for p in model.parameters(): p.requires_grad = False; model.eval()
    B_vocab_list = p7_ckpt.get("B_vocab_list", [])
    w(f"[P7-2048] loaded | {sum(p.numel() for p in model.parameters()):,} params | "
      f"vocab={len(B_vocab_list)} words | avg_cos={p7_ckpt.get('avg_cos','?')}")

    @torch.no_grad()
    def enc_word(w):
        if len(w) == 1: c1 = c2 = w[0]
        else: c1, c2 = w[0], w[-1]
        if c1 not in c2i or c2 not in c2i: return None
        return p1(torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE), last_loss=0.0)[0]

    # 构建B词表向量
    if B_vocab_list:
        B_table = torch.stack([enc_word(w) for w in B_vocab_list if enc_word(w) is not None])
        B_words = [w for w in B_vocab_list if enc_word(w) is not None]
    else:
        # Fallback: use P1 full word table
        word_table = p1.get_all_reference_vectors(DEVICE)
        B_table = F.normalize(word_table, dim=-1)
        B_words = [i2w[i] for i in range(len(i2w))]
    w(f"[B-table] {len(B_words)} words\n")

    # 测试用例
    tests = [
        (["你", "喜欢", "什么"], "object question"),
        (["谁", "教", "学生"], "subject question"),
        (["老师", "教", "什么"], "profession+object"),
        (["他", "去", "哪里"], "location question"),
        (["你", "怎么", "学习"], "adverb question"),
        (["谁", "去", "学校"], "subject+location"),
        (["我", "喜欢", "苹果"], "direct statement"),
        (["学生", "学习", "知识"], "direct statement 2"),
        (["他", "吃", "什么"], "eat+object"),
        (["你", "看", "什么"], "watch+object"),
        (["谁", "写", "作业"], "write+subject"),
        (["我", "去", "哪里"], "I+location"),
    ]

    for q_words, desc in tests:
        w(f"\n{'='*60}")
        w(f"Q: {' '.join(q_words)} ({desc})")
        w(f"{'='*60}")

        # 编码
        A_vecs = []
        for wq in q_words:
            v = enc_word(wq)
            if v is None:
                w(f"  [SKIP] '{wq}' not encodable")
                A_vecs = None; break
            A_vecs.append(v)
        if A_vecs is None: continue
        A_tensor = torch.stack(A_vecs)

        # 用B词表检索
        topk_idx, topk_scores, attn = model.retrieve_from_table(A_tensor, B_table, topk=5)
        # topk_idx: [nA, 5], topk_scores: [nA, 5]

        w(f"  [检索结果] top-5 per position:")
        for i, wq in enumerate(q_words):
            cands = [(B_words[idx.item()], f"{topk_scores[i,j].item():.3f}")
                     for j, idx in enumerate(topk_idx[i])]
            w(f"    {wq:8s} → {cands}")

        # 注意力摘要
        attn_avg = attn.mean(dim=0)  # [nA, vocab_B]
        w(f"  [注意力峰值] per A position:")
        for i, wq in enumerate(q_words):
            top_j = attn_avg[i].argmax().item()
            w(f"    {wq:8s} → {B_words[top_j]} (attn={attn_avg[i,top_j].item():.4f})")

    w(f"\n{'='*60}")
    w(f"日志: {LOG_PATH}")
    log.close()


if __name__ == "__main__":
    main()
