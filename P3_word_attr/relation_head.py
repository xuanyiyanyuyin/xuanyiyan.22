"""
P3-R 属性关系头: Q=源词, K/V=目标属性族全部词 → 交叉注意力强绑定
谓语→宾语族, 主语→谓语族, 谓语→地点族 等泛化关系
"""
import torch, torch.nn as nn, torch.nn.functional as F

class AttrRelationHead(nn.Module):
    """Q=源词向量, K/V=目标属性全部词向量 → attn自动找最匹配的"""
    def __init__(self, word_dim=128, attn_dim=128, heads=8):
        super().__init__()
        self.heads=heads; self.head_dim=attn_dim//heads; self.scale=self.head_dim**-0.5
        self.q_proj=nn.Linear(word_dim,attn_dim,bias=False)
        self.k_proj=nn.Linear(word_dim,attn_dim,bias=False)
        self.v_proj=nn.Linear(word_dim,attn_dim,bias=False)

    def forward(self, query_vec, target_pool):
        """query:[128] or [b,128], target_pool:[N,128] → attn_scores [N]"""
        if query_vec.dim()==1: query_vec=query_vec.unsqueeze(0)
        b=query_vec.shape[0]; N=target_pool.shape[0]
        q=self.q_proj(query_vec).view(b,self.heads,self.head_dim)  # [b,h,d]
        k=self.k_proj(target_pool).view(N,self.heads,self.head_dim).permute(1,0,2)  # [h,N,d]
        scores=torch.einsum('bhd,hnd->bhn',q,k)*self.scale  # [b,h,N]
        return scores.mean(dim=1).squeeze(0)  # [N]

    def topk(self, query_vec, target_pool, idx_map, k=5):
        """返回top-k词及其分数"""
        scores=self.forward(query_vec,target_pool)
        top=torch.topk(scores,min(k,len(scores)))
        return [(idx_map[i],scores[i].item()) for i in top.indices.tolist()]
