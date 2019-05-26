import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, emb_dim, ntokens):
        super().__init__()

        # TODO: pretrained embeddings
        self.emb_dim = emb_dim
        self.ntokens = ntokens
        self.emb = nn.Embedding(num_embeddings=self.ntokens, embedding_dim=self.emb_dim, padding_idx=0)

    def forward(self, ids):

        x = self.emb(ids)

        return x


class Model(nn.Module):

    def __init__(self, emb_dim, ntokens, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = EmbeddingLayer(emb_dim, ntokens)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_dense = nn.Linear(in_features=self.encoder.emb_dim, out_features=self.hidden_dim)
        self.last_dense = nn.Linear(in_features=hidden_dim, out_features=self.output_dim)

    def forward(self, ids):

        x = self.encoder(ids)
        x = torch.mean(x, dim=1)
        x = self.hidden_dense(x)
        x = torch.relu(x)
        x = self.last_dense(x)

        return x
