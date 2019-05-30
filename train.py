import torch
import torch.nn as nn
import os
import argparse
from model import Model, PretrainedModel
from data_utils import CsvDataset, load_vocab, get_tokenizer
from torch.utils.data import DataLoader
from losses import triplet_loss, margin_loss, contrastive_loss, bce
from tensorboardX import SummaryWriter
import numpy as np
import faiss
from metrics import calculate_mrr


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', default='0')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('--model', default='DAN', choices=['DAN', 'GPT', 'GPT-2'])
parser.add_argument('-l', '--loss', default='triplet', choices=['triplet', 'bce'])


def train_one_epoch(model, optimizer, dataloader, writer, epoch, device, loss_type='bce', write_steps=50):
    model.train()

    for step, ((context, context_len), (answer, answer_len)) in enumerate(dataloader, start=epoch * len(dataloader)):
        # print(context.shape, answer.shape)
        optimizer.zero_grad()
        context_embeddings = model(context.to(device))  # [batch_size, emb_size]
        answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]

        if loss_type == 'bce':
            loss = bce(context_embeddings, answer_embeddings)
        elif loss_type == 'triplet':
            loss = triplet_loss(context_embeddings, answer_embeddings)
        else:
            raise NotImplemented('No such loss')

        if step % write_steps == 0:
            print(f'Epoch = {epoch}, step = {step}, train_loss = {loss.item()}')
            write_metrics(writer, step, loss.item())
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, writer, epoch, device, loss_type='bce'):
    contexts = []
    answers = []

    model.eval()
    loss_history = []
    for (context, context_len), (answer, answer_len) in dataloader:
        context_embeddings = model(context.to(device))  # [batch_size, emb_size]
        answer_embeddings = model(answer.to(device))  # [batch_size, emb_size]

        if loss_type == 'bce':
            loss = bce(context_embeddings, answer_embeddings)
        elif loss_type == 'triplet':
            loss = triplet_loss(context_embeddings, answer_embeddings)
        else:
            raise NotImplemented('No such loss')
        loss_history.append(loss.item())

        contexts.append(context_embeddings.cpu().detach().numpy())
        answers.append(answer_embeddings.cpu().detach().numpy())

    loss_value = np.mean(loss_history)

    contexts = np.array(contexts).reshape(-1, contexts[-1].shape[-1])
    answers = np.array(answers).reshape(-1, answers[-1].shape[-1])

    emb_size = answers.shape[1]
    faiss_index = faiss.IndexFlat(emb_size)

    faiss_index.verbose = True
    faiss_index.add(answers)
    _, indexes = faiss_index.search(contexts, k=100)

    mrr = calculate_mrr(y_true=np.arange(indexes.shape[0]).reshape(-1, 1), preds=indexes)
    write_metrics(writer, epoch * len(dataloader), loss_value, mrr=mrr, prefix='eval')
    print(f'Epoch = {epoch}, step = {epoch * len(dataloader)}, eval_loss = {loss_value}, mrr = {mrr}')


def default_params():
    params = dict()
    params['batch_size'] = 128
    params['num_epochs'] = 50

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


def load_checkpoint(model, checkpoint_name, map_location=None):

    model.load_state_dict(torch.load(checkpoint_name, map_location=map_location))

    return model


def write_metrics(writer, step, loss, mrr=None, prefix='train'):
    writer.add_scalar(f'losses/{prefix}_loss', loss, step)
    if mrr is not None:
        writer.add_scalar(f'metrics/{prefix}_mrr', mrr, step)


if __name__ == '__main__':

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    params = default_params()

    ntokens = sum(1 for _ in open(os.path.join(args.data_dir, 'vocab.txt'))) + 1

    if args.model == 'DAN':
        model = Model(emb_dim=64, ntokens=ntokens, hidden_dim=32, output_dim=16).to(device)
        vocab = load_vocab(os.path.join(args.data_dir, 'vocab.txt'))
        tokenizer = None
    elif args.model in ['GPT', 'GPT-2']:
        model = PretrainedModel(model_name=args.model).to(device)
        vocab = None
        tokenizer = get_tokenizer(args.model)
    else:
        raise NotImplementedError(f'{args.model} --- no such model')

    datasets = dict()
    dataloaders = dict()

    for name in ['train', 'test']:
        shuffle = name == 'train'
        datasets[name] = CsvDataset(
            csv_path=os.path.join(args.data_dir, f'{name}.csv'),
            vocab=vocab,
            max_len=50,
            tokenizer=tokenizer
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
        train_one_epoch(model, optimizer, dataloaders['train'], writer, epoch, device, loss_type=args.loss)
        evaluate(model, dataloaders['test'], writer, epoch, device, loss_type=args.loss)
        save_checkpoint(model, os.path.join(args.model_dir, f'checkpoint_{epoch}'))
