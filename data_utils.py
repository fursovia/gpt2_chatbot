import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import OpenAIGPTTokenizer
from pytorch_pretrained_bert import GPT2Tokenizer


def vectorize(text, vocab, max_len=20):

    vectorized = []
    words = text.split()

    for i, word in enumerate(words):
        # TODO: crop from the beggining for context
        if i >= max_len:
            break

        if word in vocab:
            vectorized.append(vocab[word])
        else:
            vectorized.append(vocab['<UNK>'])
            
    if len(words) < max_len:
        vectorized.extend([vocab['<PAD>']] * (max_len - len(words)))
        
    return vectorized


def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'GPT-2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif tokenizer_name == 'GPT':
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    else:
        raise NotImplementedError(f'{tokenizer_name} -- No such tokenizer')

    return tokenizer



def load_vocab(txt_path):

    vocab = dict()
    with open(txt_path, 'r') as file:
        c = 0
        for line in file:
            vocab[line.strip()] = c
            c += 1
            
    return vocab


class CsvDataset(Dataset):
    # TODO: try keras tokenizer https://keras.io/preprocessing/text/

    def __init__(self, csv_path, vocab=None, max_len=20, tokenizer=None):
        self.data = pd.read_csv(csv_path)
        self.word2idx = vocab
        if self.word2idx is not None:
            self.idx2word = {i: word for word, i in self.word2idx.items()}

        self.context = self.data['context'].values
        self.answer = self.data['answer'].values

        self.max_len = max_len
        self.tokenizer = tokenizer

    def __getitem__(self, item):

        # TODO: change to https://torchtext.readthedocs.io/en/latest/examples.html
        if self.tokenizer is None:
            cont = vectorize(self.context[item], self.word2idx, self.max_len)
            ans = vectorize(self.answer[item], self.word2idx, self.max_len)
        else:
            cont = self.tokenizer.encode(self.context[item])
            ans = self.tokenizer.encode(self.answer[item])

        cont = torch.tensor(cont)
        ans = torch.tensor(ans)

        return cont, ans

    def __len__(self):
        return self.data.shape[0]
