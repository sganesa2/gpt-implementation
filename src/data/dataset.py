import tiktoken
import torch
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

@dataclass
class ShakespeareDataset:
    '''Tiny-shakespeare dataset for sub-word level language modelling.'''
    file_name:str
    context_size:int

    train_size:int
    val_size:int
    test_size:int

    tokenizer_model:str='gpt2'

    trainset:torch.Tensor = field(init=False)
    valset:torch.Tensor = field(init=False)
    testset:torch.Tensor = field(init=False)

    def __post_init__(self):
        data = self.encode_input()
        assert data is not None, "Data not encoded yet. Call encode_data()."

        self.trainset = data[:self.train_size]
        i = self.train_size
        self.valset = data[i:i+self.val_size]
        i+=self.val_size
        self.testset = data[i:i+self.test_size]

    def get_batch(self, split:Literal['train','val','test'], batch_size:int=32)->tuple[torch.Tensor,torch.Tensor]:
        if split=='train':
            data = self.trainset
        elif split=='val':
            data = self.valset
        elif split=='test':
            data = self.testset
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")
        
        B,T = batch_size, self.context_size

        ix = torch.randint(len(data)-self.context_size, (batch_size,))
        x = torch.stack([data[i:i+self.context_size] for i in ix])
        y = torch.stack([data[i+1:i+self.context_size+1] for i in ix])

        x,y = x.view(B*T), y.view(B*T)
        return x, y
    
    def encode_input(self)->torch.Tensor:
        with open(Path(__file__).parent.joinpath(self.file_name), 'r') as f:
            text = f.read()
        encoder = tiktoken.get_encoding(self.tokenizer_model)
        encoded_text = encoder.encode(text)
        data = torch.tensor(encoded_text, dtype = torch.float32)

        assert (self.train_size+self.val_size+self.test_size)>=len(data), "Splits sizes exceed data length!"
        return data

if __name__=="__main__":
    dataset = ShakespeareDataset(
        file_name='dataset.txt',
        context_size=8,
        train_size = 270420,
        val_size=33802,
        test_size=33803
    )
    x,y = dataset.get_batch('test')
    print(x.shape,y.shape)