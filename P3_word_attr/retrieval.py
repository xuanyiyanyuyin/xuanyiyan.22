"""
P3检索头: 输入词+属性 → 多头交叉注意力查词表 → 输出最相关词
"""
import torch, torch.nn as nn, torch.nn.functional as F

class P3RetrievalHead(nn.Module):
    """给定词和属性, 在词表中检索最相关的替代词"""
    def __init__(self, word_dim=128, attn_dim=64, heads=4):
        super().__init__()
        self.heads=heads; self.head_dim=attn_dim//heads; self.scale=self.head_dim**-0.5

        # Q: 输入词+属性嵌入 → [heads, head_dim]
        self.q_proj=nn.Linear(word_dim,attn_dim,bias=False)
        # K,V: P1词表
        self.k_proj=nn.Linear(word_dim,attn_dim,bias=False)
        self.v_proj=nn.Linear(word_dim,attn_dim,bias=False)
        self.out=nn.Linear(attn_dim,word_dim,bias=False)

    def forward(self, query_word_vec, word_table, topk=1):
        """query: [b,128], word_table: [N,128] → topk words"""
        b,N=query_word_vec.shape[0],word_table.shape[0]
        q=self.q_proj(query_word_vec).view(b,self.heads,self.head_dim)  # [b,h,d]
        k=self.k_proj(word_table).view(N,self.heads,self.head_dim).permute(1,0,2)  # [h,N,d]
        v=self.v_proj(word_table).view(N,self.heads,self.head_dim).permute(1,0,2)

        scores=torch.einsum('bhd,hnd->bhn',q,k)*self.scale  # [b,h,N]
        attn=F.softmax(scores,dim=-1)
        attn_out=torch.einsum('bhn,hnd->bhd',attn,v).contiguous().view(b,self.heads*self.head_dim)

        # 最相关词的索引
        avg_attn=attn.mean(dim=1)  # [b,N]
        top_idx=avg_attn.argmax(dim=-1)  # [b]

        # 输出: 检索到的词向量 + 索引
        retrieved_vec=self.out(attn_out)  # [b,128]
        return retrieved_vec, top_idx, attn
