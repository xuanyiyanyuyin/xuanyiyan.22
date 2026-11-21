"""
P1 字符→词语 验收测试
======================
加载模型 → 逐词评估 → 失败分析 → 注意力可视化
"""
import torch, torch.nn.functional as F, os, sys, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from P1_char_word.model import CharToWordModel


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    m = CharToWordModel(ckpt["num_chars"], ckpt["num_words"]).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


def test(model, ckpt):
    idx2char = ckpt["idx2char"]; idx2word = ckpt["idx2word"]
    nw = ckpt["num_words"]; words = [idx2word[i] for i in range(nw)]
    ref = model.get_all_reference_vectors(DEVICE)

    # 构建所有词对
    all_pairs = []
    for w in words:
        c1, c2 = ckpt["char2idx"][w[0]], ckpt["char2idx"][w[1]]
        all_pairs.append([c1, c2])

    print(f"\n{'='*60}")
    print(f"P1 字符→词语 验收测试")
    print(f"模型: epoch={ckpt.get('epoch','?')} | 词汇: {nw} | 字符: {ckpt['num_chars']}")
    print(f"注意力: {ATTN_HEADS}头×{ATTN_DIM}维 (每头{ATTN_HEAD_DIM}D)")
    print(f"{'='*60}")

    correct_top1 = 0; correct_top3 = 0; failures = []

    for i in range(nw):
        pair = torch.tensor([all_pairs[i]], device=DEVICE)
        pred, details = model(pair, last_loss=0.0, return_details=True)
        sims = F.cosine_similarity(pred, ref)
        idx = torch.argsort(sims, descending=True)
        rank = (idx == i).nonzero(as_tuple=True)[0].item()

        if rank == 0: correct_top1 += 1
        if rank < 3: correct_top3 += 1

        if rank >= 3:
            failures.append({
                "word": words[i],
                "rank": rank,
                "top3": [idx2word[j.item()] for j in idx[:3]],
                "top3_sims": [f"{sims[j].item():.3f}" for j in idx[:3]],
            })

    top1 = correct_top1 / nw; top3 = correct_top3 / nw
    print(f"\n结果:")
    print(f"  Top-1: {correct_top1}/{nw} = {top1:.2%} {'✓' if top1>=0.80 else '✗'}")
    print(f"  Top-3: {correct_top3}/{nw} = {top3:.2%} {'✓' if top3>=0.95 else '✗'}")

    if failures:
        print(f"\n失败案例 (Top-3未命中, 共{len(failures)}):")
        for f in failures[:10]:
            print(f"  '{f['word']}' rank={f['rank']} → 误认为: {f['top3']}")

    print(f"\n随机抽样 (5个词,含注意力):")
    for _ in range(5):
        i = random.randrange(nw)
        w = words[i]
        pair = torch.tensor([all_pairs[i]], device=DEVICE)
        pred, details = model(pair, last_loss=0.0, return_details=True)
        sims = F.cosine_similarity(pred, ref)
        top5 = torch.topk(sims, 5)
        cands = [(idx2word[j.item()], f"{s.item():.3f}") for j, s in zip(top5.indices, top5.values)]
        print(f"  '{w[0]}'+'{w[1]}' → '{w}'")
        print(f"    Top5: {cands}")
        aw = details["attn_weights"][0].mean(dim=0)
        print(f"    注意力: c1→c1:{aw[0,0]:.2f} c1→c2:{aw[0,1]:.2f} "
              f"c2→c1:{aw[1,0]:.2f} c2→c2:{aw[1,1]:.2f}")

    print(f"\n{'='*60}")
    print(f"验收: {'通过' if top1>=0.80 else '未通过 (需调参/加epoch/扩数据)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    for p in [os.path.join(SAVE_DIR, "P1_final.pt"), os.path.join(SAVE_DIR, "P1_best.pt")]:
        if os.path.exists(p):
            model, ckpt = load_model(p)
            test(model, ckpt)
            break
    else:
        print("错误: 未找到模型, 请先运行 train.py")
