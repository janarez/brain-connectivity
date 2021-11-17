import pickle
from functools import partial
from typing import List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from ..scripts.nolitsa import surrogates


class dotdict(dict):
    "Dictionary that allows accessing elements like attributes."
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dotdict_collate(batch):
    "Allow dot access to collated dictionary."
    elem = batch[0]
    return dotdict({key: default_collate([d[key] for d in batch]) for key in elem})


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
    return torch.from_numpy(matrix[i])


def identity_matrix(size: int, i: List[int]):
    return torch.diag(torch.ones(size)).unsqueeze(0).repeat(len(i), 1, 1)


with open("../pickles/timeseries.pickle", "rb") as f:
    ts = pickle.load(f)


def aaft_surrogates(timeseries: np.array, upsample: int):
    return _get_surrogates(timeseries, upsample, surrogates.aaft)


def iaaft_surrogates(timeseries: np.array, upsample: int, maxiter=1000, atol=1e-8, rtol=1e-10):
    """
    Upsamples each timeserie `upsample` times using the "iaaft" method.

    Note: The default parameters of `surrogates.iaaft` are taken from `nolitsa` module.
    """
    sur_func = partial(surrogates.iaaft, maxiter=maxiter, atol=atol, rtol=rtol)
    return _get_surrogates(timeseries, upsample, sur_func)[..., 0]


def _get_surrogates(timeseries, upsample, sur_func):
    samples, regions, ts_length = timeseries.shape
    # Placeholder array.
    ts_surrogates = np.empty((upsample, samples, regions, ts_length))

    # Each sample get `upsample` new timeseries for each region.
    assert upsample > 0, f"Must `upsample` by positive integer, got {upsample}."
    for sample in range(samples):
        for i in range(upsample):
            for region in range(regions):
                ts_surrogates[sample][i][region] = sur_func(timeseries[sample][region])[0]
    return ts_surrogates
