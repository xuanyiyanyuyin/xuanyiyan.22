"""
P3 7属性批量训练 (1000轮详细日志)
=================================
"""
import torch, torch.nn.functional as F, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel
from P3_word_attr.model import SubjectBindingModel, margin_loss


def gpu_mem():
    if DEVICE.type != "cuda": return (0,0,0)
    return (torch.cuda.memory_allocated(DEVICE)/1024**2, 0, 0)


def train_attr(attr_name, ep=1000, disp=50, lr=0.005):
    print(f"\n{'#'*70}")
    print(f"# P3-{attr_name} 训练启动: epochs={ep} display={disp} lr={lr}")
    print(f"{'#'*70}\n")

    # P1
    p1_path = os.path.join(SAVE_DIR, "P1_best.pt")
    p1_ckpt = torch.load(p1_path, map_location=DEVICE)
    p1 = CharToWordModel(p1_ckpt["num_chars"], p1_ckpt["num_words"]).to(DEVICE)
    p1.load_state_dict(p1_ckpt["model_state_dict"])
    for p in p1.parameters(): p.requires_grad = False; p1.eval()
    c2i = p1_ckpt["char2idx"]; i2w = p1_ckpt["idx2word"]

    # 加载词表
    data_dir = "C:/ai/P3_data"
    path = None
    for f in os.listdir(data_dir):
        if attr_name in f and f.endswith('.txt') and os.path.getsize(os.path.join(data_dir,f)) > 10:
            path = os.path.join(data_dir, f); break
    if not path:
        print(f"[跳过] 未找到{attr_name}词表"); return

    with open(path, 'r', encoding='utf-8') as f:
        attr_raw = [w.strip() for w in f if w.strip()]

    def _get_chars(w):
        if len(w)==1: return w[0], w[0]
        return w[0], w[1]

    attr_words = [w for w in attr_raw if _get_chars(w)[0] in c2i and _get_chars(w)[1] in c2i]
    attr_set = set(attr_words)
    non_attr = [w for w in i2w.values() if w not in attr_set and len(w)>=1
                and w[0] in c2i and (w[0] if len(w)==1 else w[1]) in c2i]
    print(f"[数据] {attr_name}: {len(attr_words)}词 | 非{attr_name}: {len(non_attr)}词")

    @torch.no_grad()
    def encode(words):
        vecs = []
        for w in words:
            c1, c2 = _get_chars(w)
            cids = torch.tensor([[c2i[c1], c2i[c2]]], device=DEVICE)
            vecs.append(p1(cids, last_loss=0.0)[0])
        return torch.stack(vecs) if vecs else torch.zeros(0,128,device=DEVICE)

    family_p1 = encode(attr_words)
    family_proto = family_p1.mean(dim=0)
    p3_w2id = {w: i for i, w in enumerate(i2w.values())}
    pos_ids = torch.tensor([p3_w2id[w] for w in attr_words if w in p3_w2id], device=DEVICE)
    neg_ids = torch.tensor([p3_w2id[w] for w in non_attr if w in p3_w2id], device=DEVICE)
    print(f"[P3嵌入] pos={len(pos_ids)} neg={len(neg_ids)}")

    model = SubjectBindingModel(len(p3_w2id)).to(DEVICE)
    print(f"[模型] 参数: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    last_loss = 1.0; half = BATCH_SIZE//2; total_t0 = time.time()

    for epoch in range(1, ep+1):
        t0 = time.time(); el = 0.0; nb = 0
        if len(pos_ids) > 0 and len(neg_ids) > 0:
            pi = pos_ids[torch.randperm(len(pos_ids))[:half]]
            ni = neg_ids[torch.randperm(len(neg_ids))[:half]]
            ids = torch.cat([pi, ni])
            pos = torch.tensor([True]*len(pi)+[False]*len(ni), device=DEVICE)

            for bs in range(0, len(ids), BATCH_SIZE):
                be = min(bs+BATCH_SIZE, len(ids))
                out, attn, q_raw = model(ids[bs:be], family_p1, last_loss=last_loss)
                loss, sims = margin_loss(out, family_proto, pos[bs:be], q_raw=q_raw)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item(); nb += 1; last_loss = loss.item()

        al = el/max(nb,1)
        elapsed = time.time()-t0

        if epoch % disp == 0 or epoch == 1:
            # 详细日志
            model.eval()
            with torch.no_grad():
                ps = model.binding_score(pos_ids, family_p1) if len(pos_ids)>0 else torch.tensor([0.])
                ns = model.binding_score(neg_ids, family_p1) if len(neg_ids)>0 else torch.tensor([0.])
            model.train()
            pm, nm = ps.mean().item(), ns.mean().item()
            t = (pm+nm)/2 if len(ps)>0 and len(ns)>0 else 0.5
            acc = ((ps>t).float().mean()+(ns<=t).float().mean())/2 if len(ps)*len(ns)>0 else 0

            print(f"\n{'='*70}")
            print(f"[P3-{attr_name}] Epoch {epoch:4d}/{ep} | Loss: {al:.6f} | 累计: {time.time()-total_t0:.0f}s")
            print(f"{'='*70}")
            print(f"  正({attr_name}) cos: mean={pm:.4f} std={ps.std().item():.4f} min={ps.min().item():.4f} max={ps.max().item():.4f}")
            print(f"  负(非{attr_name}) cos: mean={nm:.4f} std={ns.std().item():.4f} min={ns.min().item():.4f} max={ns.max().item():.4f}")
            print(f"  分类准确率: {acc:.2%} | 阈值: {t:.4f} | 正负差距: {pm-nm:.4f}")

            # 调制
            exp = model.explore_state
            meta = model.meta_fc(exp * min(last_loss*20,1.0))
            print(f"  探索区: mean={exp.mean().item():.6f} std={exp.std().item():.6f} range=[{exp.min().item():.4f},{exp.max().item():.4f}]")
            print(f"  元学习调制: L2={meta.norm().item():.4f}")

            # 采样5个正样本和5个负样本
            if len(pos_ids) >= 5:
                samp_p = pos_ids[:5]
                scores_p = model.binding_score(samp_p, family_p1)
                print(f"  正样本采样: {[f'{s:.3f}' for s in scores_p.tolist()]}")
            if len(neg_ids) >= 5:
                samp_n = neg_ids[:5]
                scores_n = model.binding_score(samp_n, family_p1)
                print(f"  负样本采样: {[f'{s:.3f}' for s in scores_n.tolist()]}")

            a = gpu_mem()[0]
            print(f"  GPU: {a:.0f}MB | epoch: {elapsed:.1f}s")
            print(f"{'='*70}\n")
        elif epoch % 25 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {al:.6f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        ps = model.binding_score(pos_ids, family_p1) if len(pos_ids)>0 else torch.tensor([0.])
        ns = model.binding_score(neg_ids, family_p1) if len(neg_ids)>0 else torch.tensor([0.])
    pm, nm = ps.mean().item(), ns.mean().item()
    acc = ((ps>((pm+nm)/2)).float().mean()+(ns<=((pm+nm)/2)).float().mean())/2
    print(f"\n{'#'*60}")
    print(f"# P3-{attr_name} 完成 | 正cos: {pm:.4f} | 负cos: {nm:.4f} | Acc: {acc:.2%}")
    print(f"# 验收: 正>0.7 负<0.3 Acc>95% → {'PASS' if pm>0.7 and nm<0.3 and acc>0.95 else 'FAIL'}")
    print(f"{'#'*60}")

    torch.save({"model": model.state_dict(), "attr": attr_name, "acc": acc,
                "p3_word2id": p3_w2id, "family_words": attr_words},
               os.path.join(SAVE_DIR, f"P3_{attr_name}_best.pt"))
    return pm, nm, acc


# 主流程
ATTRS = ["主语", "谓语", "宾语", "定语", "状语", "补语", "虚词"]

if __name__ == "__main__":
    results = {}
    for attr in ATTRS:
        try:
            pm, nm, acc = train_attr(attr, ep=1000, disp=50, lr=0.005)
            results[attr] = (pm, nm, acc)
        except Exception as e:
            print(f"[{attr}] 错误: {e}")

    print(f"\n{'='*70}")
    print("7属性训练总结:")
    for attr, (pm, nm, acc) in results.items():
        status = "PASS" if pm>0.7 and nm<0.3 and acc>0.95 else "FAIL"
        print(f"  {attr}: 正cos={pm:.4f} 负cos={nm:.4f} Acc={acc:.2%} {status}")
