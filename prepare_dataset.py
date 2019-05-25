import pandas as pd
import os
import argparse
import nltk
import re
from sklearn.model_selection import train_test_split
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')


def clean_sentence(text):
    text = re.sub(r'[^A-Za-z ]', '', text)
    text = text.lower().strip()
    return text


def load_data(data_path):
    data = pd.read_csv(data_path)

    if 'Label' in data.columns:
        data.drop(columns='Label', inplace=True)

    if 'Context' in data.columns and 'Utterance' in data.columns:
        data.rename(columns={'Context': 'context', 'Utterance': 'answer'}, inplace=True)

    assert 'context' in data.columns and 'answer' in data.columns

    return data


def create_vocab(corpus, min_freq=5):
    vocab_couner = Counter()

    for text in corpus:
        words = text.split()
        vocab_couner.update(words)

    popular_words = [word for word, count in vocab_couner.most_common() if count >= min_freq]
    popular_words.insert(0, '<PAD>')
    popular_words.insert(1, '<UNK>')

    vocab = {word: i for i, word in enumerate(popular_words)}

    return vocab


def save_vocab_to_txt(vocab, path):

    with open(path, 'w') as file:
        for word, _ in vocab.items():
            file.write(f'{word}\n')


if __name__ == '__main__':
    args = parser.parse_args()

    data = load_data(os.path.join(args.data_dir, 'data.csv'))  # columns = ['context', 'answer']

    data['context'] = data['context'].map(clean_sentence)
    data['answer'] = data['answer'].map(clean_sentence)

    train, test = train_test_split(data, test_size=0.1, random_state=24)

    vocab = create_vocab(train['context'].tolist() + train['answer'].tolist())

    save_vocab_to_txt(vocab, os.path.join(args.data_dir, 'vocab.txt'))
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(args.data_dir, 'test.csv'), index=False)
