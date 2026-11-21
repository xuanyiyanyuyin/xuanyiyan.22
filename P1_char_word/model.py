"""
P1 字符→词语 (v5.0 非对称: 内部2048D+512头 → 输出128D)
========================================================
编码: 2048D内部语义, 512头×2048D交叉注意力
输出: Linear(2048→128)投影 → 128D词向量 → 兼容下游P2-P9
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.config import (POS_DIM, CONTENT_DIM, CHAR_DIM, WORD_DIM,
                          P1_CONTENT_DIM, P1_CHAR_DIM, P1_HEADS, P1_HEAD_DIM, P1_MOD_DIM,
                          ATTN_HEADS, ATTN_DIM, ATTN_HEAD_DIM, ATTN_DROPOUT, MOD_DIM)


class PositionEncoding(nn.Module):
    def __init__(self, dim=POS_DIM, max_len=2):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions):
        return self.pe[positions]


class CrossAttention(nn.Module):
    """512头×2048D Q,K,V投影 → 2048D输出 (einsum, 无expand)"""
    def __init__(self, char_dim=P1_CHAR_DIM, word_dim=WORD_DIM, heads=P1_HEADS):
        super().__init__()
        self.char_dim = char_dim
        self.word_dim = word_dim
        self.heads = heads
        self.head_dim = P1_HEAD_DIM  # 4
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(char_dim * 2, char_dim, bias=False)   # 4096→2048
        self.W_k = nn.Linear(word_dim, char_dim, bias=False)        # 128→2048
        self.W_v = nn.Linear(word_dim, char_dim, bias=False)        # 128→2048
        self.W_o = nn.Linear(char_dim, char_dim, bias=False)        # 2048→2048
        self.dropout = nn.Dropout(ATTN_DROPOUT)

    def forward(self, char_vectors, word_table, return_weights=False):
        b = char_vectors.shape[0]; N = word_table.shape[0]
        q_input = torch.cat([char_vectors[:, 0, :], char_vectors[:, 1, :]], dim=-1)
        q = self.W_q(q_input).view(b, self.heads, self.head_dim)    # [b, 512, 4]
        k = self.W_k(word_table).view(N, self.heads, self.head_dim) # [N, 512, 4]
        v = self.W_v(word_table).view(N, self.heads, self.head_dim) # [N, 512, 4]
        k = k.permute(1, 0, 2)  # [512, N, 4]
        v = v.permute(1, 0, 2)  # [512, N, 4]
        attn_scores = torch.einsum('bhd,hnd->bhn', q, k) * self.scale  # [b, 512, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.einsum('bhn,hnd->bhd', attn_weights, v)       # [b, 512, 4]
        attn_out = attn_out.contiguous().view(b, self.char_dim)
        output = self.W_o(attn_out)
        if return_weights: return output, attn_weights
        return output


class PosFusion(nn.Module):
    def forward(self, pos_vectors):
        out = pos_vectors[:, 0, :] + pos_vectors[:, 1, :]
        return F.normalize(out, dim=-1)


class ExplorationZone(nn.Module):
    def __init__(self, dim=P1_MOD_DIM):
        super().__init__()
        self.pos_basis = nn.Parameter(torch.randn(dim) * 1.0)
        self.neg_basis = nn.Parameter(torch.randn(dim) * 1.0)
        self.strength_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 2), nn.Tanh())

    def forward(self, last_loss):
        lt = torch.tensor([last_loss], device=self.pos_basis.device)
        scales = self.strength_mlp(lt)
        modulation = self.pos_basis * scales[0] + self.neg_basis * scales[1]
        loss_factor = min(last_loss * 20.0, 1.0)
        return modulation * loss_factor


class MetaLearningZone(nn.Module):
    def __init__(self, dim=P1_MOD_DIM):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Identity(),
            nn.Linear(dim, dim, bias=False), nn.Identity())

    def forward(self, explore_modulation):
        return self.transform(explore_modulation)


class MLPProjection(nn.Module):
    """v5.2 轻量非线性投影 (2层GELU + 强残差)

    主路径: 2048→512(GELU)→128
    残差:   Linear(2048→128) shortcut 直通 (保持梯度)
    融合:   主路径 + 残差 → LayerNorm → 128D
    """
    def __init__(self, in_dim=2048, hidden=512, out_dim=128):
        super().__init__()
        # 主路径: 一层非线性压缩
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

        # 残差 shortcut: 直通保底
        self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
        self.ln_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        # 主路径: 非线性特征重组
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.fc2(h)

        # 残差 shortcut: 线性直通
        s = self.shortcut(x)

        # 融合
        return self.ln_out(h + s)


class CharToWordModel(nn.Module):
    """P1 v5.2: 2048D内部编码 + MLP非线性投影 → 128D词向量输出"""
    def __init__(self, num_chars, num_words):
        super().__init__()
        self.num_chars = num_chars
        self.num_words = num_words

        # 内部高维编码
        self.char_content = nn.Embedding(num_chars, P1_CONTENT_DIM)  # 2040D
        nn.init.xavier_uniform_(self.char_content.weight)
        self.pos_encoder = PositionEncoding()

        # 512头交叉注意力: Q=字符对(4096D), K,V=词表(128D) → 2048D输出
        self.cross_attn = CrossAttention()

        # 调制链 (2048D)
        self.explore_zone = ExplorationZone()
        self.meta_zone = MetaLearningZone()

        # MLP非线性投影: 2048→512→128 + 残差shortcut
        self.output_proj = MLPProjection(2048, 512, 128)

        # 词表: 128D (与下游统一)
        self.word_table = nn.Parameter(torch.randn(num_words, WORD_DIM) * 0.01)

    def get_char_vectors(self, char_ids):
        b = char_ids.shape[0]
        positions = torch.arange(2, device=char_ids.device).unsqueeze(0).repeat(b, 1)
        pos = self.pos_encoder(positions)              # [b, 2, 8]
        content = self.char_content(char_ids)          # [b, 2, 2040]
        full = torch.cat([pos, content], dim=-1)       # [b, 2, 2048]
        return pos, content, full

    def project_char(self, char_internal):
        """将内部2048D字符向量投影到128D输出空间"""
        return F.normalize(self.output_proj(char_internal), dim=-1)

    def forward(self, char_ids, last_loss=1.0, return_details=False):
        pos, content, full = self.get_char_vectors(char_ids)

        # 交叉注意力: Q(2048D内部), K,V(128D词表) → 2048D
        attn_out, attn_weights = self.cross_attn(full, self.word_table, return_weights=True)

        # 调制链 (2048D)
        explore_mod = self.explore_zone(last_loss)
        meta_mod = self.meta_zone(explore_mod)
        modulated_attn = attn_out + meta_mod.unsqueeze(0).expand(pos.shape[0], -1)

        # 位置融合: 8D
        pos_out = self.pos_fusion(pos)

        # 语义融合: 2040D内部 → 投影到120D
        base_sem = content[:, 0, :] + content[:, 1, :]            # [b, 2040]
        attn_sem = modulated_attn[:, POS_DIM:]                     # [b, 2040]
        internal_sem = F.normalize(base_sem + attn_sem, dim=-1)    # [b, 2040]

        # 拼接内部向量: [b, 2048]
        internal_vec = torch.cat([pos_out, internal_sem], dim=-1)

        # 分阶段投影: 2048D → 1024D(*/门控) → 128D(+-融合)
        word_vector = self.project_char(internal_vec)              # [b, 128]

        if return_details:
            return word_vector, {
                "pos_raw": pos, "content_raw": content, "char_full": full,
                "attn_weights": attn_weights, "attn_out_raw": attn_out,
                "modulated_attn": modulated_attn,
                "explore_mod": explore_mod, "meta_mod": meta_mod,
                "pos_out": pos_out, "content_out": internal_sem,
            }
        return word_vector

    def get_word_target(self, word_ids):
        return self.word_table[word_ids]

    @torch.no_grad()
    def get_all_reference_vectors(self, device):
        self.eval()
        return self.word_table.clone().to(device)

    @torch.no_grad()
    def predict_word_ids(self, char_ids):
        self.eval()
        _, _, full = self.get_char_vectors(char_ids)
        _, attn_weights = self.cross_attn(full, self.word_table, return_weights=True)
        avg_attn = attn_weights.squeeze(2).mean(dim=1) if attn_weights.dim() > 3 else attn_weights.mean(dim=1)
        return avg_attn.argmax(dim=-1), avg_attn

    # PosFusion (复用, 输出8D位置)
    def pos_fusion(self, pos):
        out = pos[:, 0, :] + pos[:, 1, :]
        return F.normalize(out, dim=-1)
