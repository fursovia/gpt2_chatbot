import pandas as pd
import os
import argparse
import nltk
import re
from sklearn.model_selection import train_test_split


def clean_sentence(text):
    text = re.sub(r'[^A-Za-z ]', '', text)
    text = text.lower().strip()
    return text


def load_data(data_path):
    data = pd.read_csv(data_path)

    return data


def create_vocab(corpus):
    vocab = dict()
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    for text in corpus:
        words = text.split()
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def save_vocab_to_txt(vocab, path):

    with open(path, 'w') as file:
        for word, _ in vocab.items():
            file.write(f'{word}\n')


DATA_PATH = 'data/sample.csv'


if __name__ == '__main__':

    data = load_data(DATA_PATH)  # columns = ['context', 'answer']

    data['context'] = data['context'].map(clean_sentence)
    data['answer'] = data['answer'].map(clean_sentence)

    train, test = train_test_split(data, test_size=0.1, random_state=24)

    vocab = create_vocab(train['context'].tolist() + train['answer'].tolist())

    save_vocab_to_txt(vocab, 'data/vocab.txt')
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
