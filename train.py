import torch
import torch.nn as nn
import os
import argparse
from model import Model
from data_utils import CsvDataset
from torch.utils.data import DataLoader

def train_one_epoch(model, dataset):
    model.train()
    pass


def evaluate(model, dataset):
    model.eval()
    pass


def default_params():
    params = dict()
    params['emb_dim'] = 300
    params['batch_size'] = 128
    params['num_epochs'] = 10

    return params


def save_checkpoint(model, checkpoint_name):

    pass


if __name__ == '__main__':

    params = default_params()
    model = Model(emb_dim=params['emb_dim'])

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
