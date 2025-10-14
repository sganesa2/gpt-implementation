import torch
import math
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

from tokenizer import ToyGPTTokenizer

device = 'cpu'#'mps' if torch.backends.mps.is_available() else 'cpu'

@dataclass
class Dataset:
    '''Class for creating a dataset for sub-word level language modelling.'''
    file_name:str
    context_size:int
    tokenizer:ToyGPTTokenizer

    train_frac:float
    val_frac:int
    test_frac:int

    trainset:torch.Tensor = field(init=False)
    valset:torch.Tensor = field(init=False)
    testset:torch.Tensor = field(init=False)

    def __post_init__(self):
        data = self.encode_input()
        assert data is not None, "Data not encoded yet. Call encode_data()."

        f = lambda x: math.floor(x*len(data))
        i = f(self.train_frac)
        self.trainset = data[:i]#.to(device)
        j = f(self.val_frac)
        self.valset = data[i:i+j]#.to(device)
        k= f(self.test_frac)
        self.testset = data[i+j:i+j+k]#.to(device)
    
    def decode(self, idx:torch.Tensor)->list[str]:
        idx = idx.tolist()
        assert isinstance(idx[0], list), "Batch dimension not present in idx!"
        return [self.tokenizer.decode(inp) for inp in idx]
    
    def get_batch(self, split:Literal['train','val','test'], batch_size:int=32)->tuple[torch.Tensor,torch.Tensor]:
        if split=='train':
            data = self.trainset
        elif split=='val':
            data = self.valset
        elif split=='test':
            data = self.testset
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

        ix = torch.randint(len(data)-self.context_size, (batch_size,))
        x = torch.stack([data[i:i+self.context_size] for i in ix])#.to(device)
        y = torch.stack([data[i+1:i+self.context_size+1] for i in ix])#.to(device)
        return x, y
    
    def encode_input(self)->torch.Tensor:
        with open(Path(__file__).parent.joinpath(self.file_name), 'r') as f:
            text = f.read()
        encoded_text = self.tokenizer.encode(text)
        data = torch.tensor(encoded_text, dtype = torch.float32)
        return data

if __name__=="__main__":
    with open(Path(__file__).parent.joinpath("dataset.txt"),'r') as f:
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
    x,y = dataset.get_batch('test')
    print(x.shape,y.shape)