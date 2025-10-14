from pathlib import Path

from data.dataset import Dataset, device
from model.model import ToyGPT
from data.tokenizer import ToyGPTTokenizer

from checkpointing import load_model
from train import (
    TEST_SIZE, 
    MAX_NEW_TOKENS ,VOCAB_SIZE ,CONTEXT_SIZE, 
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC
)

def run_toygpt(dataset:Dataset, model:ToyGPT):
    checkpointer_dict = load_model()
    model_state_dict = checkpointer_dict['model_state_dict']
    model.load_state_dict(model_state_dict)
    
    inp,_ = dataset.get_batch('test', TEST_SIZE)
    out_tensor = model.generate(inp, MAX_NEW_TOKENS)
    print(dataset.decode(out_tensor)[0])
    # #model.to(device=device)

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
    model = ToyGPT(tokenizer.vocab_size)
    run_toygpt(dataset, model)