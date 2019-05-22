import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


def vectorize(text):
    return text


class CsvDataset(Dataset):

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __getitem__(self, item):

        x, y = None, None

        return x, y

    def __len__(self):
        return self.data.shape[0]
