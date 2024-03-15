import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import Dataset, DataLoader

# Dataset

class EPData(Dataset):
    def __init__(self, X, y=None):
        super().__init__()

        self.X = X
        self.y = y
    
    def __getitem__(self, ind):

        if self.y is None:
            return torch.Tensor(self.X[ind])

        else:
            return torch.Tensor(self.X[ind]), torch.Tensor(self.y[ind])

    def __len__(self):
        return len(self.X)
