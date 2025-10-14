import torch

from data.dataset import Dataset, device
from model.model import ToyGPT

def run_toygpt(file_name:str, context_size:int, train_size:int, val_size:int, test_size:int):
    dataset = Dataset(
        file_name=file_name,
        context_size=context_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    model = ToyGPT(
        vocab_size=dataset.vocab_size
    )
    # #model.to(device=device)
    # inps, _ = dataset.get_batch('test', batch_size = 1)
    # idx = model(inps)
    # print(dataset.decode(idx)[0])

if __name__=="__main__":
    run_toygpt(
        file_name = "dataset.txt",
        context_size = 8,
        train_size = 270420,
        val_size=33802,
        test_size=33803
    )