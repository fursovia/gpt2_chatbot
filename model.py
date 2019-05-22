import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, text):
        return None
