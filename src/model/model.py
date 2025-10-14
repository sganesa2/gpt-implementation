import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyGPT(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.generator = torch.Generator().manual_seed(6385189022)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size, dtype=torch.float64)

    def forward(self, inps:torch.Tensor, targets:torch.Tensor=None, reg_factor:float=0.1)->tuple[torch.Tensor, torch.Tensor]:
        logits = self.token_embedding_table(inps)
        if targets is None: return logits, None
        assert reg_factor, "Provide regularization factor"

        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets, label_smoothing=reg_factor)
        return logits,loss
    
    def generate(self, inps:torch.Tensor, max_new_tokens:int)->torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(inps)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1, generator = self.generator)
            inps = torch.cat((inps,idx), dim=1)
        return inps
