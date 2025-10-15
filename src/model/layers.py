import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class Head(nn.Module):
    def __init__(self, context_size:int, n_embed:int, head_size:int):
        super().__init__()
        self.head_size = n_embed
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((context_size,context_size))))
    
    def forward(self, inps:torch.Tensor)->torch.Tensor:
        _,T,_ = inps.shape

        q,k,v = self.query(inps), self.key(inps), self.value(inps)
        att_scores = (q @ k.transpose(-2,-1)) * self.head_size**-0.5
        att_scores = torch.masked_fill(att_scores, mask=self.tril[:T,:T]==0, value=-float('inf'))
        att_weights = F.softmax(att_scores, dim=-1)
        out = att_weights@v
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads:int, context_size:int, n_embed:int):
        super().__init__()
        self.heads = nn.ModuleList([Head(context_size, n_embed, n_embed//num_heads) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, inps:torch.Tensor)->torch.Tensor:
        return torch.cat([h(inps) for h in self.heads], -1)
    
class FeedForward(nn.Module):
    def __init__(self, n_embed:int, proj_factor:int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, n_embed*proj_factor),
            nn.ReLU(),
            nn.Linear(n_embed*proj_factor, n_embed)
        )
    
    def forward(self, inps:torch.Tensor)->torch.Tensor:
        return self.mlp(inps)

class Block(nn.Module):
    def __init__(self,num_heads:int, context_size:int, n_embed:int, proj_factor:int):
        super().__init__()
        self.multi_head_att = MultiHeadedAttention(num_heads,context_size,n_embed)
        self.mlp = FeedForward(n_embed, proj_factor)

    def forward(self, inps:torch.Tensor)->torch.Tensor:
        inps = inps+self.multi_head_att(inps)
        inps = inps + self.mlp(inps)
        return inps