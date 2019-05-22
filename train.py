import torch
import torch.nn as nn
import os
import argparse
from model import Model, EmbeddingLayer
from data_utils import CsvDataset
from torch.utils.data import DataLoader
from loss import triplet_loss, margin_loss, contrastive_loss


def train_one_epoch(model, optimizer, dataloader):
    model.train()

    for context, answer in dataloader:
        optimizer.zero_grad()
        context_embeddings = model(context)  # [batch_size, emb_size]
        answer_embeddings = model(answer)  # [batch_size, emb_size]

        loss = triplet_loss(context_embeddings, answer_embeddings)
        loss.backward()

        optimizer.step()


def evaluate(model, dataloader):
    model.eval()
    pass


def default_params():
    params = dict()
    params['emb_dim'] = 300
    params['batch_size'] = 128
    params['num_epochs'] = 10

    return params


def model_params():
    params = dict()

    params['loss'] = 'triples'
    params['sampling'] = 'uniform'

    return params


def save_checkpoint(model, checkpoint_name):

    pass





if __name__ == '__main__':

    params = default_params()

    ntokens = 100  # len(vocab) FIX!
    emb_encoder = EmbeddingLayer(emb_dim=300, ntokens=ntokens)
    model = Model(encoder=emb_encoder)

    datasets = dict()
    dataloaders = dict()
    for name in ['train', 'test']:
        shuffle = name == 'train'
        datasets[name] = CsvDataset(csv_path=f'{name}.csv')
        dataloaders[name] = DataLoader(datasets[name], batch_size=params['batch_size'], shuffle=shuffle)

    for epoch in range(params['num_epochs']):
        train_one_epoch(model, datasets['train'])
        evaluate(model, datasets['test'])

        save_checkpoint(model, f'checkpoint_{epoch}')
