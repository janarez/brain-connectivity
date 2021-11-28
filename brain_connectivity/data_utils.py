import os
import pickle
from functools import partial
from typing import List

import numpy as np
import torch
from nolitsa import surrogates
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from .general_utils import get_logger


class dotdict(dict):
    "Dictionary that allows accessing elements like attributes."
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dotdict_collate(batch):
    "Allow dot access to collated dictionary."
    elem = batch[0]
    return dotdict(
        {key: default_collate([d[key] for d in batch]) for key in elem}
    )


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


def iaaft_surrogates(
    timeseries: np.array, upsample: int, maxiter=1000, atol=1e-8, rtol=1e-10
):
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
                ts_surrogates[sample][i][region] = sur_func(
                    timeseries[sample][region]
                )[0]
    return ts_surrogates


class StratifiedCrossValidation:
    def __init__(
        self,
        log_folder,
        targets,
        num_assess_folds=10,
        num_select_folds=10,
        random_state=42,
    ):
        self.targets = targets
        self.num_assess_folds = num_assess_folds
        self.num_select_folds = num_select_folds

        self._outer_skf = StratifiedKFold(
            n_splits=num_assess_folds, random_state=random_state, shuffle=True
        )
        self._inner_skf = StratifiedKFold(
            n_splits=num_select_folds, random_state=random_state, shuffle=True
        )

        # Iterator over outter CV: assessment of model performance.
        self._outer_cv_iterator = enumerate(
            self._outer_skf.split(np.empty(shape=len(targets)), targets)
        )

        self.logger = get_logger("cv", os.path.join(log_folder, "cv.txt"))

    def outter_cross_validation(self):
        """
        Generates new stratified folds and updates test and dev indices.
        """
        while True:
            try:
                i, (dev_split, test_split) = next(self._outer_cv_iterator)
            except StopIteration:
                break

            self.dev_indices, self.test_indices = dev_split, test_split
            self._set_inner_cv_iterator()
            self.logger.info(f"Outer fold {i+1} / {self.num_assess_folds}")
            yield i

    def _set_inner_cv_iterator(self):
        self._inner_cv_iterator = enumerate(
            self._inner_skf.split(
                np.empty(shape=len(self.dev_indices)),
                self.targets[self.dev_indices],
            )
        )

    def inner_cross_validation(self):
        """
        Generates new stratified folds and updates val and train indices.
        """
        while True:
            try:
                i, (train_split, val_split) = next(self._inner_cv_iterator)
            except StopIteration:
                # Reset for next experiment.
                self._set_inner_cv_iterator()
                break

            self.train_indices, self.val_indices = (
                self.dev_indices[train_split],
                self.dev_indices[val_split],
            )
            self.logger.info(f"Inner fold {i+1} / {self.num_select_folds}")
            yield i
