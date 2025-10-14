import torch
from dataclasses import dataclass
from pathlib import Path

from data.dataset import Dataset
from model.model import ToyGPT
from data.tokenizer import ToyGPTTokenizer

@dataclass
class ToyGPTTrainer:
    dataset: Dataset
    model:ToyGPT

    def train(self, iters:int, lr:float=1e-3, reg_factor:float=0.1)->None:
        optim = torch.optim.Adam(self.model.parameters(), lr)
        for _ in range(iters):
            inps, targets = self.dataset.get_batch("train")
            optim.zero_grad()
            _, loss = self.model.forward(inps, targets, reg_factor)
            loss.backward()
            optim.step()

    @torch.no_grad()
    def estimate_loss(self, eval_iters:int, batch_size:int=32, reg_factor:float=0.1)->dict:
        out = {}
        self.model.eval()
        for split in ['train','val']:
            evaluated_loss = torch.zeros(eval_iters)
            for i in range(eval_iters):
                inps, targets = self.dataset.get_batch(split, batch_size)
                _,loss = self.model(inps,targets, )
                evaluated_loss[i] = loss.item()
            out[split] = evaluated_loss.mean(0)
        self.model.train()
        return out
    
if __name__=="__main__":
    with open(Path(__file__).parent.joinpath("data/dataset.txt"),'r') as f:
        text = f.read()
    tokenizer = ToyGPTTokenizer(948)
    tokenizer.train(text)
    dataset = Dataset(
        file_name='dataset.txt',
        context_size=8,
        tokenizer=tokenizer,
        train_frac = 0.8,
        val_frac=0.1,
        test_frac=0.1
    )
    model = ToyGPT(tokenizer.vocab_size)
    trainer = ToyGPTTrainer(dataset, model)
    trainer.train(2000)
    loss = trainer.estimate_loss(300)
    print(loss)
    inp,_ = dataset.get_batch('test', 1)
    out_tensor = model.generate(inp, 100)
    print(dataset.decode(out_tensor)[0])