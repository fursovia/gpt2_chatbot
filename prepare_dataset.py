import pandas as pd
import os
import argparse
import nltk
import re
from sklearn.model_selection import train_test_split
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-mf', '--min_freq', type=int, default=5)


def clean_sentence(text):

    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
    text = " ".join(text.split())

    return text


def load_data(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()

    if 'Label' in data.columns:
        data.drop(columns='Label', inplace=True)

    if 'Context' in data.columns and 'Utterance' in data.columns:
        data.rename(columns={'Context': 'context', 'Utterance': 'answer'}, inplace=True)

    assert 'context' in data.columns and 'answer' in data.columns

    data = data[(data['context'].map(len) > 0) & (data['answer'].map(len) > 0)]

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

    print('Cleaning the data ...')
    data['original_answers'] = data['answer']
    data['context'] = data['context'].map(clean_sentence)
    data['answer'] = data['answer'].map(clean_sentence)

    data = data[(data['context'].map(len) > 0) & (data['answer'].map(len) > 0)]

    train, test = train_test_split(data[~data['answer'].duplicated()], test_size=0.1, random_state=24)

    vocab = create_vocab(data['context'].tolist() + data['answer'].tolist(), min_freq=args.min_freq)

    print(f'Vocab size = {len(vocab)}')

    save_vocab_to_txt(vocab, os.path.join(args.data_dir, 'vocab.txt'))
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(args.data_dir, 'test.csv'), index=False)
