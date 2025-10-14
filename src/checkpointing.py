import torch

from model.model import ToyGPT

def save_model(epoch:int, model:ToyGPT, evaluated_loss:dict, file_name:str = 'checkpoint.pth')->None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': evaluated_loss,
    }
    torch.save(checkpoint, file_name)

def load_model(file_name:str='checkpoint.pth')->dict:
    checkpoint = torch.load(file_name)
    return checkpoint


def main(inp:int):
    print("Hello from gpt-implementation!")
    return inp+1

if __name__ == "__main__":
    print(main(0))
