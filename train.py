import torch
import torch.nn as nn
import os
import argparse
from model import Model


def train_one_epoch(model, dataset):
    pass


def evaluate(model, dataset):
    pass


def default_params():
    params = dict()
    params['emb_dim'] = 300

    return params


if __name__ == '__main__':

    params = default_params()
    model = Model(emb_dim=params['emb_dim'])

    train_one_epoch(model, )