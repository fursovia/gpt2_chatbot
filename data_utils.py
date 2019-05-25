import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


def vectorize(text, vocab, max_len=20):

    vectorized = []
    words = text.split()

    for i, word in enumerate(words):
        if i >= max_len:
            break

        if word in vocab:
            vectorized.append(vocab[word])
        else:
            vectorized.append(vocab['<UNK>'])
            
    if len(words) < max_len:
        vectorized.extend([vocab['<PAD>']]*(max_len - len(words)))
        
    return vectorized


def load_vocab(txt_path):

    vocab = dict()
    with open(txt_path, 'r') as file:
        c = 0
        for line in file:
            vocab[line.strip()] = c
            c += 1
            
    return vocab


class CsvDataset(Dataset):

    def __init__(self, csv_path, vocab_path, max_len=20):
        self.data = pd.read_csv(csv_path)
        self.vocab = load_vocab(vocab_path)

        self.context = self.data['context'].values
        self.answer = self.data['answer'].values

        self.max_len = max_len

    def __getitem__(self, item):

        # TODO: change to https://torchtext.readthedocs.io/en/latest/examples.html
        cont = vectorize(self.context[item], self.vocab, self.max_len)
        ans = vectorize(self.answer[item], self.vocab, self.max_len)

        cont = torch.tensor(cont)
        ans = torch.tensor(ans)

        return cont, ans

    def __len__(self):
        return self.data.shape[0]