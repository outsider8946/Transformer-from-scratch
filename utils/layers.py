import torch
import math
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_f: int, out_f: int, masked: bool):
        super().__init__()
        self.q = nn.Linear(in_features=in_f, out_features=out_f)
        self.k = nn.Linear(in_features=in_f, out_features=out_f)
        self.v = nn.Linear(in_features=in_f, out_features=out_f)
        self.masked = masked

    def forward(self, x: torch.Tensor, K: Optional[torch.Tensor] = None, V: Optional[torch.Tensor] = None):
        if K is None and V is None:
            Q = self.q(x)
        else:
            Q = x
        if K is None:
            K = self.k(x)
        if V is None:
            V = self.v(x)
        assert K is not None and V is not None
    
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))

        if self.masked:
            size = (attention_weights.shape[1], attention_weights.shape[2])
            mask = torch.triu(torch.zeros(size=size, device=attention_weights.device)*float('-inf'))
            attention_weights += mask

        attention_weights = F.softmax(attention_weights, dim=-1)

        return torch.matmul(attention_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int, d_model:int, d_k:int, masked: bool):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.masked = masked

        self.WO = nn.Linear(in_features=num_heads*d_k, out_features=d_model)
        self.WQ = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_k) for _ in range(num_heads)])
        self.WK = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_k) for _ in range(num_heads)])
        self.WV = nn.ModuleList([nn.Linear(in_features=d_model, out_features=d_k) for _ in range(num_heads)])
    
    def forward(self, x:torch.Tensor, Q: Optional[torch.Tensor] = None, K: Optional[torch.Tensor] = None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        encoder_ouput = Q is not None and K is not None
        heads = []

        for Wq, Wk, Wv in zip(self.WQ, self.WK, self.WV):
            if encoder_ouput:
                Q = Wq(Q)
                K = Wk(K)
            else:
                Q = Wq(x)
                K = Wq(x)

            V = Wv(x)
            heads.append(Attention(in_f=self.d_model, out_f=self.d_k, masked=self.masked)(Q, K, V))
        
        return self.WO(torch.cat(heads, dim=-1)) 


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_f1: int, out_f1: int, in_f2: int, out_f2: int):
        super().__init__()
        self.l1 = nn.Linear(in_features=in_f1, out_features=out_f1)
        self.l2 = nn.Linear(in_features=in_f2, out_features=out_f2)
    
    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        return x


class AddNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return F.layer_norm(input=x1+x2, normalized_shape=[self.d_model])