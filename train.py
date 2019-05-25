import torch
import torch.nn as nn
import os
import argparse
from model import Model, EmbeddingLayer
from data_utils import CsvDataset, load_vocab
from torch.utils.data import DataLoader
from losses import triplet_loss, margin_loss, contrastive_loss
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', default='0')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')


def train_one_epoch(model, optimizer, dataloader, writer, epoch, device, write_steps=50):
    model.train()

    for step, (context, answer) in enumerate(dataloader, start=epoch * len(dataloader)):
        # print(context.shape, answer.shape)
        optimizer.zero_grad()
        context_embeddings = model(context.to(device))  # [batch_size, emb_size]
        answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]

        loss = triplet_loss(context_embeddings, answer_embeddings)
        # loss = bce(context_embeddings, answer_embeddings)
        if step % write_steps == 0:
            print(f'Epoch = {epoch}, step = {step}, train_loss = {loss.item()}')
            write_metrics(writer, step, loss.item())
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, writer, epoch, device):
    model.eval()

    for step, (context, answer) in enumerate(dataloader, start=epoch * len(dataloader)):
        context_embeddings = model(context.to(device))  # [batch_size, emb_size]
        answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]

        loss = triplet_loss(context_embeddings, answer_embeddings)
        # loss = bce(context_embeddings, answer_embeddings)
        write_metrics(writer, step, loss.item(), prefix='eval')
        print(f'Epoch = {epoch}, step = {step}, eval_loss = {loss.item()}')


def default_params():
    params = dict()
    params['emb_dim'] = 300
    params['batch_size'] = 64
    params['num_epochs'] = 500

    return params


# def model_params():
#     params = dict()
#
#     params['loss'] = 'triples'
#     params['sampling'] = 'uniform'
#     params['emb_dim'] = 300
#     params['output_dim'] = 64
#
#     return params


def save_checkpoint(model, checkpoint_name):

    torch.save(model.state_dict(), checkpoint_name)


def load_checkpoint(model, checkpoint_name):

    model.load_state_dict(torch.load(checkpoint_name))

    return model


def write_metrics(writer, step, loss, prefix='train'):
    writer.add_scalar(f'losses/{prefix}_loss', loss, step)


if __name__ == '__main__':

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    params = default_params()

    ntokens = sum(1 for _ in open(os.path.join(args.data_dir, 'vocab.txt'))) + 1
    model = Model(emb_dim=64, ntokens=ntokens, output_dim=32).to(device)

    datasets = dict()
    dataloaders = dict()
    vocab = load_vocab(os.path.join(args.data_dir, 'vocab.txt'))
    for name in ['train', 'test']:
        shuffle = name == 'train'
        datasets[name] = CsvDataset(
            csv_path=os.path.join(args.data_dir, f'{name}.csv'),
            vocab=vocab,
            max_len=20
        )

        dataloaders[name] = DataLoader(
            datasets[name],
            batch_size=params['batch_size'],
            shuffle=shuffle,
            drop_last=True
        )

    writer = SummaryWriter(args.model_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for epoch in range(params['num_epochs']):
        train_one_epoch(model, optimizer, dataloaders['train'], writer, epoch, device)
        evaluate(model, dataloaders['test'], writer, epoch, device)
        save_checkpoint(model, os.path.join(args.model_dir, f'checkpoint_{epoch}'))
