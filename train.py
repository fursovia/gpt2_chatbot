import torch
import torch.nn as nn
import os
import argparse
from model import Model, EmbeddingLayer
from data_utils import CsvDataset
from torch.utils.data import DataLoader
from loss import triplet_loss, margin_loss, contrastive_loss
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', default='0')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')


def train_one_epoch(model, optimizer, dataloader, writer, epoch, device):
    model.train()

    for step, (context, answer) in enumerate(dataloader, start=epoch * len(dataloader)):
        optimizer.zero_grad()
        context_embeddings = model(context.to(device))  # [batch_size, emb_size]
        answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]

        loss = triplet_loss(context_embeddings, answer_embeddings)
        write_metrics(writer, step, loss.item())
        loss.backward()

        optimizer.step()


def evaluate(model, dataloader, writer, epoch, device):
    model.eval()

    with torch.no_grad():
        for step, (context, answer) in enumerate(dataloader, start=epoch * len(dataloader)):
            context_embeddings = model(context.to(device))  # [batch_size, emb_size]
            answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]
            loss = triplet_loss(context_embeddings, answer_embeddings)
            write_metrics(writer, step, loss.item(), prefix='eval')


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

    torch.save(model.state_dict(), checkpoint_name)


def load_checkpoint(model, checkpoint_name):

    model.load_state_dict(torch.load(checkpoint_name))

    return model


def write_metrics(writer, step, loss, prefix='train'):
    writer.add_scalar(f'{prefix}_loss', loss, step)


if __name__ == '__main__':

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = default_params()

    ntokens = sum(1 for _ in open(os.path.join(args.data_dir, 'vocab.txt'))) + 1
    model = Model(emb_dim=300, ntokens=ntokens, output_dim=64).to(device)

    datasets = dict()
    dataloaders = dict()
    for name in ['train', 'test']:
        shuffle = name == 'train'
        datasets[name] = CsvDataset(csv_path=f'{name}.csv')
        dataloaders[name] = DataLoader(datasets[name], batch_size=params['batch_size'], shuffle=shuffle)

    writer = SummaryWriter(args.model_dir)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(params['num_epochs']):
        train_one_epoch(model, optimizer, dataloaders['train'], writer, epoch, device)
        evaluate(model, dataloaders['test'], writer, epoch, device)

        save_checkpoint(model, os.path.join(args.model_dir, f'checkpoint_{epoch}'))
