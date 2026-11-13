"""
P7 跨句词级路由 (128D + 64头 + 2048头词级)
===========================================
CrossSentenceRouter: 64头句子级路由 → B句向量
P7WordRouter2048: 2048头词级交叉注意力 → 每A词→B词映射
Q=A句词, K=B句词(或全词表), V=B句词, head_dim=1极细粒度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossSentenceRouter(nn.Module):
    def __init__(self, word_dim=128, attn_dim=256, sent_dim=256, heads=64):
        super().__init__()
        self.word_dim = word_dim
        self.attn_dim = attn_dim
        self.sent_dim = sent_dim
        self.heads = heads
        self.head_dim = attn_dim // heads  # 4
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256
        self.k_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256
        self.v_proj = nn.Linear(word_dim, attn_dim, bias=False)  # 128→256

        self.sent_fuse = nn.Linear(attn_dim, sent_dim, bias=False)  # 256→256

        self.explore_state = nn.Parameter(torch.randn(sent_dim) * 1.0)
        self.meta_fc = nn.Sequential(
            nn.Linear(sent_dim, sent_dim, bias=False))

    def forward(self, A_word_vecs, B_word_table, last_loss=1.0):
        nA = A_word_vecs.shape[0]
        nB = B_word_table.shape[0]

        q = self.q_proj(A_word_vecs).view(nA, self.heads, self.head_dim)
        k = self.k_proj(B_word_table).view(nB, self.heads, self.head_dim)
        v = self.v_proj(B_word_table).view(nB, self.heads, self.head_dim)

        q = q.transpose(0, 1)  # [heads, nA, 4]
        k = k.transpose(0, 1)  # [heads, nB, 4]
        v = v.transpose(0, 1)  # [heads, nB, 4]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)  # [heads, nA, 4]

        attn_out = attn_out.transpose(0, 1).contiguous().view(nA, self.attn_dim)  # [nA, 256]
        sent_raw = attn_out.mean(dim=0)  # [256]
        sent_vec = self.sent_fuse(sent_raw)  # [256]

        loss_factor = min(last_loss * 20.0, 1.0)
        mod = self.meta_fc(self.explore_state * loss_factor)
        sent_vec = sent_vec + mod

        return sent_vec, attn  # [256], [heads, nA, nB]


class P7WordRouter2048(nn.Module):
    """32头×4维交叉注意力 + 层级调制(探索→元学习→主代码)

    架构流程:
      输入: A_word_vecs[nA,128] + B_word_vecs[nB,128]
      1. 位置编码加法注入(pos_weight可学习)
      2. Q/K/V投影: 128→128 (Q/K xavier, V eye)
      3. 32头注意力: head_dim=4, 每头4维点积(4倍区分度)
         scores[32,nA,nB] = Q[h,:4]·K[h,:4] * scale * temp
      4. 层级调制:
         探索区: loss→GELU MLP→64D控制信号
         元学习区: 探索信号+meta_state→词级gate[128D]+句级gate[256D]
         主代码: word_out *= word_gate, sent_vec += sent_mod * sent_gate
      5. 输出: word_out[nA,128], sent_vec[256], attn[32,nA,nB]
              + word_gate[128], sent_gate[256]

    参数: ~224K (32头×4dim + 探索/元学习MLP)
    """
    def __init__(self, word_dim=128, inner_dim=128, heads=32, head_dim=4, max_len=32,
                 num_groups=4, sent_dim=256):
        super().__init__()
        self.word_dim = word_dim          # 128
        self.inner_dim = inner_dim        # 128
        self.heads = heads                # 32
        self.head_dim = head_dim          # 4
        self.scale = self.head_dim ** -0.5
        self.num_groups = num_groups
        self.heads_per_group = heads // num_groups  # 8

        # 输入投影: A侧(p7_input_proj) → 内部维度 (word_dim≠inner_dim时)
        self.need_proj = (word_dim != inner_dim)
        if self.need_proj:
            self.input_proj = nn.Linear(word_dim, inner_dim, bias=False)
            nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)

        # 位置编码: inner_dim维强信号
        self.register_buffer('pos_pe', self._build_sinusoidal_pe(max_len, inner_dim))
        self.pos_weight = nn.Parameter(torch.tensor(1.0))
        self.pos_embed = nn.Parameter(torch.randn(max_len, inner_dim) * 0.5)

        # Q/K投影: 128→128
        self.q_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)

        # V独立头: 每头独立4×128投影
        self.v_heads = nn.Parameter(torch.randn(heads, head_dim, inner_dim) * 0.02)

        self.q_norm = nn.RMSNorm(inner_dim)  # Q/K模长归一化, 防连乘爆炸
        self.k_norm = nn.RMSNorm(inner_dim)

        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 语义组偏好: 4组×4维
        self.group_bias = nn.Parameter(torch.eye(num_groups, head_dim) * 0.5)

        # 输出投影: 16→128 (4组×4维=16)
        self.out_proj = nn.Linear(num_groups * head_dim, word_dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

        # 句向量投影: 128→256
        self.sent_proj = nn.Linear(word_dim, sent_dim, bias=False)  # 128→sent_dim
        # P5式±叠加: 每位置可学习权重(可正可负), 替代mean
        self.sent_pos_w = nn.Parameter(torch.ones(max_len))

        # ═══════════════════════════════════════
        # 层级调制: 探索区→元学习区→主代码
        # ═══════════════════════════════════════

        # ── 探索区: loss(标量) → 控制信号(发给元学习区) ──
        # 12D全量loss→64D信号 (lossless传输, 每个loss独立编码)
        self.explore_net = nn.Sequential(
            nn.Linear(12, 48, bias=True),
            nn.GELU(),
            nn.Linear(48, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 64, bias=True),
        )
        self.explore_state = nn.Parameter(torch.randn(64) * 0.02)

        # ── 元学习区: 探索信号+自身状态 → 词级/句级门控 ──
        self.meta_state = nn.Parameter(torch.randn(128) * 0.02)

        # 词级门控: GELU防死区
        self.meta_word_gate = nn.Sequential(
            nn.Linear(128 + 64, 96, bias=True),
            nn.GELU(),
            nn.Linear(96, word_dim, bias=True),
        )

        # 句级门控
        self.meta_sent_gate = nn.Sequential(
            nn.Linear(128 + 64, 96, bias=True),
            nn.GELU(),
            nn.Linear(96, sent_dim, bias=True),
        )
        self.meta_sent_mod = nn.Sequential(
            nn.Linear(128 + 64, 96, bias=True),
            nn.GELU(),
            nn.Linear(96, sent_dim, bias=False),
        )

        self.gate_sharpness = nn.Parameter(torch.tensor(1.0))  # 软门控: 连续无极调节

    def _build_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, A_word_vecs, B_word_vecs, last_loss=1.0):
        """v9: V独立头 + 语义分组竞争 + 组间软隔离"""
        nA = A_word_vecs.shape[0]
        nB = B_word_vecs.shape[0]
        device = A_word_vecs.device

        pos = self.pos_pe[:nA].to(device)
        A_with_pos = A_word_vecs + pos * self.pos_weight + self.pos_embed[:nA]

        q = self.q_norm(self.q_proj(A_with_pos))
        k = self.k_norm(self.k_proj(B_word_vecs))
        hd = self.head_dim
        q = q.view(nA, self.heads, hd).transpose(0, 1)  # [32, nA, 4]
        k = k.view(nB, self.heads, hd).transpose(0, 1)  # [32, nB, 4]

        # V独立头
        v = torch.einsum('hdw,bw->bhd', self.v_heads, B_word_vecs)  # [nB, 32, 4]
        v = v.transpose(0, 1)  # [32, nB, 4]

        temp_raw = F.softplus(self.temperature) + 0.1
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale * temp_raw
        attn = F.softmax(scores, dim=-1)  # [32, nA, nB]
        attn_out = torch.matmul(attn, v)  # [32, nA, 4]

        # ── 语义分组竞争: 4组×8头, 组内softmax选最佳头 ──
        attn_out_g = attn_out.view(self.num_groups, self.heads_per_group, nA, hd)  # [4, 8, nA, 4]
        gb = self.group_bias.unsqueeze(1).unsqueeze(2)  # [4, 1, 1, 4]
        group_scores = (attn_out_g * gb).sum(dim=-1)  # [4, 8, nA]
        group_weights = F.softmax(group_scores / 0.5, dim=1)  # T=0.5软选择, 防组内坍缩
        group_out = (attn_out_g * group_weights.unsqueeze(-1)).sum(dim=1)  # [4, nA, 4]
        group_out = group_out.transpose(0, 1).contiguous().view(nA, self.num_groups * hd)  # [nA, 16]

        word_out = self.out_proj(group_out)  # [nA, 128]

        # ═══════════════════════════════════════
        # 层级调制: 探索区→元学习区→主代码
        # ═══════════════════════════════════════
        # 10D全量loss→64D信号 (lossless传输)
        # 无外部loss_vec时退化为标量模式
        if hasattr(self, '_loss_vec') and self._loss_vec is not None:
            loss_input = self._loss_vec.to(device).unsqueeze(0)  # [1, 10]
        else:
            loss_factor = min(last_loss * 20.0, 1.0)
            loss_input = torch.full((1, 12), loss_factor, device=device)

        # ── 探索区: 10D loss → 控制信号(64D) ──
        explore_signal = self.explore_net(loss_input)  # [1, 64]

        # 保底噪声: 防脑死亡 (signal_norm永不为0)
        if self.training:
            explore_signal = explore_signal + torch.randn_like(explore_signal) * 0.01

        explore_signal = explore_signal + self.explore_state  # 去掉*loss_factor衰减

        # ── 元学习区: 探索信号 + 自身状态 → 门控 ──
        meta_input = torch.cat([self.meta_state.unsqueeze(0),
                                explore_signal], dim=-1)  # [1, 128+64] 去掉*loss_factor

        # 词级软门控 → 管理主代码输出
        word_gate = torch.sigmoid(self.meta_word_gate(meta_input) * self.gate_sharpness)  # [1, 128]
        word_out = word_out * word_gate  # 逐维门控主代码

        # 句向量: P5式±叠加 (替代mean, 保留词级信息)
        pw = self.sent_pos_w[:nA].unsqueeze(1)  # [nA, 1]
        sent_vec = self.sent_proj((word_out * pw).sum(dim=0))  # [256]
        sent_gate = torch.sigmoid(self.meta_sent_gate(meta_input) * self.gate_sharpness)  # [1, 256]
        sent_mod = self.meta_sent_mod(meta_input)  # [1, 256]
        sent_vec = sent_vec + sent_mod.squeeze(0) * sent_gate.squeeze(0)

        return word_out, sent_vec, attn, word_gate, sent_gate

    def forward_batch(self, A_padded, B_padded, a_lens, b_lens, last_loss=1.0):
        """批量forward: A_padded[bs, max_nA, 128], B_padded[bs, max_nB, 128]"""
        bs, max_nA, _ = A_padded.shape
        _, max_nB, _ = B_padded.shape
        device = A_padded.device

        # 位置编码 (广播到batch)
        pos = self.pos_pe[:max_nA].unsqueeze(0).to(device)  # [1, max_nA, 128]
        A_with_pos = A_padded + pos * self.pos_weight + self.pos_embed[:max_nA].unsqueeze(0)

        q = self.q_norm(self.q_proj(A_with_pos)).view(bs, max_nA, self.heads, self.head_dim).permute(0,2,1,3)
        k = self.k_norm(self.k_proj(B_padded)).view(bs, max_nB, self.heads, self.head_dim).permute(0,2,1,3)

        v = torch.einsum('hdx,bsx->bshd', self.v_heads, B_padded)
        v = v.permute(0,2,1,3)

        temp_raw = F.softplus(self.temperature) + 0.1
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale * temp_raw  # [bs,32,max_nA,max_nB]

        # Padding mask
        mask_a = torch.arange(max_nA, device=device).unsqueeze(0) < torch.tensor(a_lens, device=device).unsqueeze(1)  # [bs, max_nA]
        mask_b = torch.arange(max_nB, device=device).unsqueeze(0) < torch.tensor(b_lens, device=device).unsqueeze(1)  # [bs, max_nB]
        attn_mask = mask_a.unsqueeze(1).unsqueeze(-1) * mask_b.unsqueeze(1).unsqueeze(2)  # [bs, 1, max_nA, max_nB]
        scores = scores.masked_fill(~attn_mask, -6e4)  # FP16安全(-65504)

        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)  # [bs, 32, max_nA, 4]

        attn_out_g = attn_out.view(bs, self.num_groups, self.heads_per_group, max_nA, self.head_dim)
        gb = self.group_bias.view(1, self.num_groups, 1, 1, self.head_dim)
        group_scores = (attn_out_g * gb).sum(dim=-1)  # [bs, 4, 8, max_nA]
        group_weights = F.softmax(group_scores / 0.5, dim=2)
        group_out = (attn_out_g * group_weights.unsqueeze(-1)).sum(dim=2)  # [bs, 4, max_nA, 4]
        group_out = group_out.permute(0,2,1,3).contiguous().view(bs, max_nA, self.num_groups * self.head_dim)

        word_out_raw = self.out_proj(group_out)  # [bs, max_nA, 128]

        # 层级调制 (用上一轮的_loss_vec, 若无则zeros)
        if hasattr(self, '_loss_vec') and self._loss_vec is not None:
            loss_input = self._loss_vec.to(device).unsqueeze(0).expand(bs, -1)
        else:
            loss_input = torch.zeros(bs, 12, device=device)
        explore_signal = self.explore_net(loss_input)  # [bs, 64]
        explore_signal = explore_signal + self.explore_state.unsqueeze(0)
        meta_input = torch.cat([self.meta_state.unsqueeze(0).expand(bs, -1), explore_signal], dim=-1)
        word_gate = torch.sigmoid(self.meta_word_gate(meta_input) * self.gate_sharpness)  # [bs, 128]
        word_out = word_out_raw * word_gate.unsqueeze(1)

        # sent_vec
        pw = self.sent_pos_w[:max_nA].view(1, max_nA, 1).to(device)
        mask_a_f = mask_a.float().unsqueeze(-1)
        sent_vec = self.sent_proj((word_out * pw * mask_a_f).sum(dim=1))
        sent_gate_raw = torch.sigmoid(self.meta_sent_gate(meta_input) * self.gate_sharpness)
        sent_mod = self.meta_sent_mod(meta_input)
        sent_vec = sent_vec + sent_mod * sent_gate_raw

        return word_out, sent_vec, attn, word_gate, sent_gate_raw

    def get_group_losses(self, A_word_vecs, B_word_vecs):
        """软惩罚: 组间隔离 + 反聚焦熵 (辩论方案: 阈值软约束)"""
        nA = A_word_vecs.shape[0]; nB = B_word_vecs.shape[0]
        device = A_word_vecs.device

        pos = self.pos_pe[:nA].to(device)
        A_with_pos = A_word_vecs + pos * self.pos_weight + self.pos_embed[:nA]
        q = self.q_proj(A_with_pos).view(nA, self.heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(B_word_vecs).view(nB, self.heads, self.head_dim).transpose(0, 1)
        v = torch.einsum('hdw,bw->bhd', self.v_heads, B_word_vecs).transpose(0, 1)

        temp_raw = F.softplus(self.temperature) + 0.1
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale * temp_raw
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out_g = attn_out.view(self.num_groups, self.heads_per_group, nA, self.head_dim)

        # 组间软隔离: 只惩罚cos>0.3的组间相似 (允许合法关联)
        group_avg = attn_out_g.mean(dim=1).mean(dim=2)  # [4, 4]
        gn = F.normalize(group_avg, dim=-1)
        cos_mat = torch.mm(gn, gn.T)  # [4,4]
        mask = 1.0 - torch.eye(self.num_groups, device=device)
        group_pen = F.relu((cos_mat * mask).abs() - 0.3).mean()

        # 反聚焦熵惩罚: 防止每头注意力太平均
        attn_clamped = attn.clamp(min=1e-8)
        entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(nB, dtype=torch.float, device=device))
        norm_entropy = entropy / max_entropy
        entropy_pen = F.relu(norm_entropy - 0.5).mean()

        return group_pen, entropy_pen

    def retrieve_from_table(self, A_word_vecs, B_table, topk=5):
        """推理模式: 补上门控逻辑, 消除训练/推理不一致"""
        self.eval()
        nA = A_word_vecs.shape[0]; nB = B_table.shape[0]
        device = A_word_vecs.device
        pos = self.pos_pe[:nA].to(device)
        A_with_pos = A_word_vecs + pos * self.pos_weight + self.pos_embed[:nA]

        hd = self.head_dim
        q = self.q_proj(A_with_pos).view(nA, self.heads, hd).transpose(0, 1)
        k = self.k_proj(B_table).view(nB, self.heads, hd).transpose(0, 1)
        v = torch.einsum('hdw,bw->bhd', self.v_heads, B_table).transpose(0, 1)

        temp_soft = F.softplus(self.temperature) + 0.1
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale * temp_soft
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)

        # 组内竞争 (与forward一致)
        attn_out_g = attn_out.view(self.num_groups, self.heads_per_group, nA, hd)
        gb = self.group_bias.unsqueeze(1).unsqueeze(2)
        group_scores = (attn_out_g * gb).sum(dim=-1)
        group_weights = F.softmax(group_scores / 0.5, dim=1)
        group_out = (attn_out_g * group_weights.unsqueeze(-1)).sum(dim=1)
        group_out = group_out.transpose(0, 1).contiguous().view(nA, self.num_groups * hd)

        word_out = self.out_proj(group_out)

        # 补上门控 (之前缺失!) — 推理时 loss=0, 门控为静态基线
        loss_tensor = torch.tensor([[0.0]], device=device)
        explore_signal = self.explore_net(loss_tensor)
        meta_input = torch.cat([torch.zeros(1, 128, device=device), explore_signal], dim=-1)
        word_gate = torch.sigmoid(self.meta_word_gate(meta_input) * self.gate_sharpness)
        word_out = word_out * word_gate

        word_out_n = F.normalize(word_out, dim=-1)
        B_table_n = F.normalize(B_table, dim=-1)
        cos_sim = torch.mm(word_out_n, B_table_n.T)
        top = torch.topk(cos_sim, topk, dim=-1)
        return top.indices, top.values, attn
