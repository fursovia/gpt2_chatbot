import torch
import torch.nn as nn
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model
from data_utils import PAD_VALUE


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

    def forward(self, ids, true_len=None):

        x = self.encoder(ids)
        x = torch.mean(x, dim=1)
        x = self.hidden_dense(x)
        x = torch.relu(x)
        x = self.last_dense(x)

        return x


class PretrainedModel(nn.Module):

    def __init__(self, model_name, add_dense=True, trainable=False):
        super().__init__()

        self.model_name = model_name
        self.add_dense = add_dense
        self.trainable = trainable

        if self.model_name == 'GPT':
            self.encoder = OpenAIGPTModel.from_pretrained('openai-gpt')
        elif self.model_name == 'GPT-2':
            self.encoder = GPT2Model.from_pretrained('gpt2')
        else:
            raise NotImplementedError(f'{self.model_name} -- No such model')

        if not self.trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if self.add_dense:
            self.dense = nn.Linear(in_features=768, out_features=128)

    def forward(self, ids, true_len=None):

        if self.model_name == 'GPT':
            output = self.encoder(ids)
        elif self.model_name == 'GPT-2':
            output, _ = self.encoder(ids)
        else:
            raise NotImplementedError(f'{self.model_name} -- No such model')

        output = torch.masked_fill(output, ids.unsqueeze(-1) == PAD_VALUE, 0)
        output = torch.mean(output, dim=1)

        if self.add_dense:
            output = self.dense(output)

        return output
