from typing import List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class dotdict(dict):
    "Dictionary that allows accessing elements like attributes."
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DenseDataset(Dataset):
    "Simple dictionary dataset."

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


def zeroth_axis_sample(matrix: np.array, i: List[int]):
    return torch.from_numpy(matrix[i]).to(torch.float32)


def identity_matrix(size: int, i: List[int]):
    return torch.diag(torch.ones(size)).unsqueeze(0).repeat(len(i), 1, 1)
