import pickle
import re
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

@dataclass
class ToyGPTTokenizer:
    vocab_size:int
    
    token_to_id:dict = field(init=False, default=None)
    id_to_token:dict = field(init=False, default=None)
    merges:dict = field(init=False, default_factory=lambda:dict())
    path:Path = field(init=False, default = Path(__file__).parent.joinpath("tokenizer.pkl"))

    def __post_init__(self):
        if self.path.exists(): 
            with open(self.path,'rb') as f:
                obj = pickle.load(f)
                self.token_to_id, self.id_to_token, self.merges = obj.get('token_to_id'), obj.get('id_to_token'), obj.get('merges',{})

    def _get_pairs(self, words:dict)->dict:
        pairs = {}
        for w,freq in words.items():
            subwords = w.split(" ")
            for i in range(len(subwords)-1):
                p = subwords[i], subwords[i+1]
                pairs[p] = pairs.get(p,0) + freq
        return pairs
    
    def _get_updated_words(self, p:tuple[str], words:dict)->dict:
        updated_words = {}
        for w, freq in words.items():
            w = w.replace(" ".join(p),"".join(p))
            updated_words[w] = freq
        return updated_words
    
    def _tokenize_word(self, word:str)->list[str]:
        word = " ".join([*word,'</w>'])
        while True:
            subwords = word.split(" ")
            if len(subwords)<1: break
            idx = self.vocab_size #we use this to find the min merge index
            pairs = [(subwords[i],subwords[i+1]) for i in range(len(subwords)-1)]

            earliest_merge = None
            for p in pairs:
                i = self.merges.get(p, float('inf'))
                if i<idx: 
                    earliest_merge = p
                    idx = i
            if not earliest_merge: break

            word = word.replace(" ".join(earliest_merge), "".join(earliest_merge))
        return word.split(" ")

    def _encoded_token(self, token:str)->list[int]:
        try:
            id = self.token_to_id.get(token)
            if id is not None: return [id]
            return [self.token_to_id[c] for c in token]
        except Exception:
            raise

    def train(self, text:str)->None:
        if self.token_to_id and self.id_to_token and self.merges: return

        vocab = set(text+'</w>')
        vocab.add('</w>')
        words = {}
        for w in text.split(" "):
            w = re.sub(' +',"",w)
            mod_word = " ".join([*w,'</w>'])
            words[mod_word] = words.get(mod_word,0)+1

        total_merges = self.vocab_size-len(vocab)
        for i in range(total_merges):
            pairs = self._get_pairs(words)
            if not pairs:break
            most_freq_pair = max(pairs, key=pairs.get)
            words = self._get_updated_words(most_freq_pair, words)
            vocab.add("".join(most_freq_pair))
            self.merges[most_freq_pair] = i
        
        self.token_to_id = {v:i for i,v in enumerate(vocab)}
        self.id_to_token = {i:v for v,i in self.token_to_id.items()}
        with open(self.path,'wb') as f:
            obj = {"token_to_id":self.token_to_id, "id_to_token":self.id_to_token, "merges":self.merges}
            pickle.dump(obj,f)
    
    def encode(self, text:str)->list[int]:
        words = text.split(" ")
        token_ids = []
        for w in words:
            w = re.sub(' +',"",w)
            tokens = self._tokenize_word(w)
            [token_ids.append(self._encoded_token(t)) for t in tokens]
        return list(chain.from_iterable(token_ids))
    
    def decode(self, token_ids:list[int])->str:
        tokens = [self.id_to_token[id] for id in token_ids]
        out = "".join(tokens).replace("</w>"," ")
        return out
    
if __name__=="__main__":
    with open(Path(__file__).parent.joinpath("dataset.txt"),'r') as f:
        text = f.read()
    tokenizer = ToyGPTTokenizer(5000)
    tokenizer.train(text)
    ids = tokenizer.encode("i will attend CMU")
    print(ids)
    print(tokenizer.decode(ids))
