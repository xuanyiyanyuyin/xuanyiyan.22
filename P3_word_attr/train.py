"""
P3 词语→属性族绑定 通用训练
=============================
支持: 主语/谓语/宾语/定语/状语/补语
P1架构: Q=P3嵌入, K,V=属性族P1向量 → 调制 → Q+调制输出
"""
import torch, torch.nn.functional as F, time, os, sys, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.model import SubjectBindingModel, margin_loss


def settings():
    attr = input("属性类型 (主语/谓语/宾语/定语/状语/补语) [默认:谓语]: ").strip() or "谓语"
    try:
        ep = int(input("epochs(60): ").strip() or "60")
        disp = int(input("display(6): ").strip() or "6")
        lr = float(input("lr(0.005): ").strip() or "0.005")
    except: ep, disp, lr = 60, 6, 0.005
    print(f"属性={attr} epochs={ep} display={disp} lr={lr}\n")
    return attr, ep, disp, lr


def load_data(attr_name):
    """加载属性词表, 自动匹配文件"""
    data_dir = "C:/ai/P3_data"
    # 映射表
    name_map = {"谓语": "pred", "主语": "subj", "宾语": "obj", "定语": "attr", "状语": "adv", "补语": "comp"}
    # 候选文件名
    candidates = [f"words_{attr_name}.txt"]
    if attr_name in name_map:
        candidates.append(f"words_{name_map[attr_name]}.txt")
    # 模糊匹配: 扫描目录找包含关键词的文件
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.txt') and f not in candidates:
                candidates.append(f)
    # 尝试打开
    for fname in candidates:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 10:
            with open(path, "r", encoding="utf-8") as f:
                words = [w.strip() for w in f if w.strip()]
            print(f"[数据] {fname} -> {len(words)}词")
            return words, True
    # 全失败: 用第一个非空的txt文件
    print(f"[警告] 未找到属性词表, 使用现有数据文件")
    return [], True


@torch.no_grad()
def evaluate(model, pos_ids, neg_ids, family_wv):
    model.eval()
    ps = model.binding_score(pos_ids, family_wv) if len(pos_ids) > 0 else torch.tensor([])
    ns = model.binding_score(neg_ids, family_wv) if len(neg_ids) > 0 else torch.tensor([])
    model.train()
    if len(ps) > 0 and len(ns) > 0:
        t = (ps.mean() + ns.mean()) / 2
        acc = ((ps > t).float().mean() + (ns <= t).float().mean()) / 2
        return {"pos_mean": ps.mean().item(), "neg_mean": ns.mean().item(),
                "accuracy": acc.item(), "threshold": t.item()}
    return {"pos_mean": 0, "neg_mean": 0, "accuracy": 0, "threshold": 0.5}


def train():
    attr, ep, disp, lr = settings()

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    p1_c2i = p1_ckpt["char2idx"]
    p1_i2w = p1_ckpt["idx2word"]

    # 属性族词表 (支持单字词: "是"→编码为["是","是"])
    attr_words, _ = load_data(attr)
    def _get_chars(w):
        if len(w) == 1: return w[0], w[0]   # 单字词重复
        return w[0], w[1]
    # 过滤: 只保留字符都在P1表中的词(单字词重复字符)
    valid_attr = []
    for w in attr_words:
        c1, c2 = _get_chars(w)
        if c1 in p1_c2i and c2 in p1_c2i:
            valid_attr.append(w)
    attr_words = valid_attr
    attr_set = set(attr_words)
    non_attr = [w for w in p1_i2w.values() if w not in attr_set]

    print(f"[数据] {attr}族: {len(attr_words)}词 | 非{attr}: {len(non_attr)}词")

    # 编码 (兼容单字词)
    @torch.no_grad()
    def encode(words):
        vecs = []
        for w in words:
            c1, c2 = _get_chars(w)
            cids = torch.tensor([[p1_c2i[c1], p1_c2i[c2]]], device=DEVICE)
            vecs.append(p1(cids, last_loss=0.0)[0])
        return torch.stack(vecs)

    family_p1 = encode(attr_words)
    family_proto = family_p1.mean(dim=0)

    # P3嵌入映射
    p3_w2id = {w: i for i, w in enumerate(p1_i2w.values())}
    num_p3 = len(p3_w2id)
    pos_ids = torch.tensor([p3_w2id[w] for w in attr_words if w in p3_w2id], device=DEVICE)
    neg_ids = torch.tensor([p3_w2id[w] for w in non_attr if w in p3_w2id], device=DEVICE)
    print(f"[P3嵌入] {num_p3}词 | pos={len(pos_ids)} neg={len(neg_ids)}")

    # 模型
    model = SubjectBindingModel(num_p3).to(DEVICE)
    print(f"[P3] 参数: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    last_loss = 1.0; best_acc = 0.0; half = BATCH_SIZE // 2

    for epoch in range(1, ep + 1):
        t0 = time.time(); epoch_loss = 0.0; nb = 0
        p_idx = pos_ids[torch.randperm(len(pos_ids))[:half]]
        n_idx = neg_ids[torch.randperm(len(neg_ids))[:half]]
        ids = torch.cat([p_idx, n_idx])
        pos = torch.tensor([True]*len(p_idx) + [False]*len(n_idx), device=DEVICE)

        for bs in range(0, len(ids), BATCH_SIZE):
            be = min(bs + BATCH_SIZE, len(ids))
            bid, bpos = ids[bs:be], pos[bs:be]
            out, attn, q_raw = model(bid, family_p1, last_loss=last_loss)
            loss, sims = margin_loss(out, family_proto, bpos, q_raw=q_raw)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); nb += 1
            last_loss = loss.item()

        elapsed = time.time() - t0

        if epoch % disp == 0 or epoch == 1 or epoch == ep:
            m = evaluate(model, pos_ids, neg_ids, family_p1)
            print(f"[P3-{attr}] Epoch {epoch:4d} | Loss: {epoch_loss/nb:.4f}")
            print(f"  正({attr}) cos: {m['pos_mean']:.3f} | 负(非{attr}) cos: {m['neg_mean']:.3f} "
                  f"| Acc: {m['accuracy']:.2%} | {elapsed:.1f}s")
            if m['accuracy'] > best_acc:
                best_acc = m['accuracy']
                torch.save({"epoch": epoch, "model": model.state_dict(), "acc": best_acc,
                            "attr": attr, "p3_word2id": p3_w2id, "family_words": attr_words},
                           os.path.join(SAVE_DIR, f"P3_{attr}_best.pt"))
        else:
            print(f"  Epoch {epoch:4d} | Loss: {epoch_loss/nb:.4f} | {elapsed:.1f}s")

    m = evaluate(model, pos_ids, neg_ids, family_p1)
    print(f"\n{'#'*60}")
    print(f"# P3-{attr}完成 | 正cos: {m['pos_mean']:.3f} | 负cos: {m['neg_mean']:.3f} | Acc: {m['accuracy']:.2%}")
    print(f"# 验收: 正>0.7 负<0.3 Acc>95% → "
          f"{'PASS' if m['pos_mean']>0.7 and m['neg_mean']<0.3 and m['accuracy']>0.95 else 'FAIL'}")
    print(f"{'#'*60}")
    torch.save({"epoch": ep, "model": model.state_dict(), "attr": attr,
                "p3_word2id": p3_w2id, "family_words": attr_words, "metrics": m},
               os.path.join(SAVE_DIR, f"P3_{attr}_final.pt"))


if __name__ == "__main__":
    train()
