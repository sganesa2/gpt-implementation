import torch
from dataclasses import dataclass
from pathlib import Path

from data.dataset import Dataset
from model.model import ToyGPT
from data.tokenizer import ToyGPTTokenizer

from checkpointing import save_model

EPOCHS = 10000
EVAL_ITERS = 300
TEST_SIZE = 1
MAX_NEW_TOKENS = 500
VOCAB_SIZE = 2000
CONTEXT_SIZE = 8
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
LR, REG_FACTOR = 3e-3, 0.1
N_EMBED, N_HEADS, N_BLOCKS, PROJ_FACTOR=64,4,4,4
DROPOUT_PROB = 0.2

@dataclass
class ToyGPTTrainer:
    dataset: Dataset
    model:ToyGPT

    def train(self, iters:int, lr:float, reg_factor:float)->None:
        optim = torch.optim.Adam(self.model.parameters(), lr)
        for _ in range(iters):
            inps, targets = self.dataset.get_batch("train")
            optim.zero_grad()
            _, loss = self.model.forward(inps, targets, reg_factor)
            loss.backward()
            optim.step()

    @torch.no_grad()
    def estimate_loss(self, eval_iters:int, batch_size:int=32)->dict:
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
    tokenizer = ToyGPTTokenizer(VOCAB_SIZE)
    tokenizer.train(text)
    dataset = Dataset(
        file_name='dataset.txt',
        context_size=CONTEXT_SIZE,
        tokenizer=tokenizer,
        train_frac = TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC
    )
    model = ToyGPT(VOCAB_SIZE, CONTEXT_SIZE, N_EMBED,N_HEADS, N_BLOCKS, PROJ_FACTOR, DROPOUT_PROB)
    trainer = ToyGPTTrainer(dataset, model)
    trainer.train(EPOCHS, LR, REG_FACTOR)
    loss = trainer.estimate_loss(EVAL_ITERS)
    print(loss)
    inp,_ = dataset.get_batch('test', TEST_SIZE)
    out_tensor = model.generate(inp, MAX_NEW_TOKENS)
    print(dataset.decode(out_tensor)[0])

    save_model(EPOCHS,model,loss)