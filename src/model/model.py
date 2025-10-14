import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import Block

class ToyGPT(nn.Module):
    def __init__(self, vocab_size:int, context_size:int, n_embed:int, n_heads:int, n_layers:int, n_blocks:int):
        super().__init__()
        self.generator = torch.Generator().manual_seed(6385189022)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.n_embed = n_embed
        #Layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(n_heads,context_size,n_embed, n_layers)
                for _ in range(n_blocks)
            ]
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, inps:torch.Tensor, targets:torch.Tensor=None, reg_factor:float=0.1)->tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.token_embedding_table(inps)
        pos_emb = self.position_embedding_table(torch.arange(0,self.context_size))

        inp_emb = token_emb+pos_emb
        blocks_op = self.blocks(inp_emb)
        logits = self.lm_head(blocks_op)

        if targets is None: return logits, None
        assert reg_factor, "Provide regularization factor"

        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets, label_smoothing=reg_factor)
        return logits,loss
    
    def generate(self, inps:torch.Tensor, max_new_tokens:int)->torch.Tensor:
        for _ in range(max_new_tokens):
            short_inp = inps[:,-self.context_size:]
            logits, _ = self(short_inp)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1, generator = self.generator)
            inps = torch.cat((inps,idx), dim=1)
        return inps
