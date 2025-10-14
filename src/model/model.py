import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyGPT(nn.Module):
    def __init__(self, vocab_size:int):
        super().__init__()
        self.generator = torch.Generator().manual_seed(6385189022)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inps:torch.Tensor, targets:torch.Tensor=None)->tuple[torch.Tensor, torch.Tensor]:
        logits = self.token_embedding_table(inps)
        if not targets: return logits, None

        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = logits.view(B*T)

        loss = F.cross_entropy(logits, targets, label_smoothing=0.1)
        return logits,loss
    
    def generate(self, inps:torch.Tensor, max_new_tokens:int)->torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(inps)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1, generator = self.generator)
            inps = torch.cat((inps,idx), dim=1)
        return inps
        
