"""
P3 属性绑定 (128D + 64头交叉注意力)
====================================
Q = P3独立词嵌入(128D)
K,V = P1家族向量(128D)
64头交叉注意力 → 属性调制 → 输出 = Q + 调制(128D)

Loss: 余弦margin + 身份保留
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class SubjectBindingModel(nn.Module):
    def __init__(self, num_words, word_dim=128, attn_dim=256, heads=64):
        super().__init__()
        self.heads = heads
        self.head_dim = attn_dim // heads  # 4
        self.scale = self.head_dim ** -0.5

        self.p3_embed = nn.Embedding(num_words, word_dim)
        nn.init.xavier_uniform_(self.p3_embed.weight)

        self.word_q = nn.Linear(word_dim, attn_dim, bias=False)
        self.family_k = nn.Linear(word_dim, attn_dim, bias=False)
        self.family_v = nn.Linear(word_dim, attn_dim, bias=False)
        self.mod_proj = nn.Linear(attn_dim, word_dim, bias=False)

        self.explore_state = nn.Parameter(torch.randn(word_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(word_dim, word_dim, bias=False))

    def forward(self, word_ids, family_p1_vecs, last_loss=1.0):
        word_ids = word_ids.long()
        b, N = word_ids.shape[0], family_p1_vecs.shape[0]
        q_raw = self.p3_embed(word_ids)

        q = self.word_q(q_raw).view(b, self.heads, self.head_dim)  # [b, h, d]
        k = self.family_k(family_p1_vecs).view(N, self.heads, self.head_dim)  # [N, h, d]
        v = self.family_v(family_p1_vecs).view(N, self.heads, self.head_dim)  # [N, h, d]
        k = k.permute(1, 0, 2)  # [h, N, d]
        v = v.permute(1, 0, 2)  # [h, N, d]
        scores = torch.einsum('bhd,hnd->bhn', q, k) * self.scale  # [b, h, N]
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.einsum('bhn,hnd->bhd', attn, v)  # [b, h, d]
        attn_out = attn_out.contiguous().view(b, self.heads * self.head_dim)
        mod = self.mod_proj(attn_out)

        loss_factor = min(last_loss * 20.0, 1.0)
        meta = self.meta_fc(self.explore_state * loss_factor)
        mod = mod + meta.unsqueeze(0).expand(b, -1)

        out = q_raw + mod
        return out, attn, q_raw

    @torch.no_grad()
    def binding_score(self, word_ids, family_p1_vecs):
        self.eval()
        out, _, _ = self.forward(word_ids, family_p1_vecs, last_loss=0.0)
        proto = family_p1_vecs.mean(dim=0, keepdim=True)
        return F.cosine_similarity(out, proto.expand(out.shape[0], -1), dim=-1)

    @torch.no_grad()
    def get_modulated(self, word_ids, family_p1_vecs):
        self.eval()
        out, _, _ = self.forward(word_ids, family_p1_vecs, last_loss=0.0)
        return out

    @torch.no_grad()
    def get_prototype(self, family_p1_vecs):
        return family_p1_vecs.mean(dim=0)


def margin_loss(modulated_vec, family_proto, is_positive, q_raw=None, margin=0.3, id_weight=0.2):
    sim = F.cosine_similarity(modulated_vec, family_proto.unsqueeze(0).expand(modulated_vec.shape[0], -1), dim=-1)
    pos = is_positive; neg = ~is_positive
    pos_l = F.relu(1.0 - margin - sim[pos]).mean() if pos.any() else 0.0
    neg_l = F.relu(sim[neg] - margin).mean() if neg.any() else 0.0

    id_l = 0.0
    if q_raw is not None and pos.any():
        id_sim = F.cosine_similarity(modulated_vec[pos], q_raw[pos], dim=-1)
        id_l = (1.0 - id_sim).mean()

    return pos_l + neg_l + id_weight * id_l, sim.detach()
