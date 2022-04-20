from functools import partial
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from nolitsa import surrogates
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset


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


def one(size: int, i: List[int]):
    return torch.ones(len(i), size, 1)


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
    return _get_surrogates(timeseries, upsample, sur_func)


def _get_surrogates(timeseries, upsample, sur_func):
    samples, regions, ts_length = timeseries.shape
    # Placeholder array.
    ts_surrogates = np.empty((samples, upsample, regions, ts_length))

    # Each sample get `upsample` new timeseries for each region.
    assert upsample > 0, f"Must `upsample` by positive integer, got {upsample}."
    for sample in range(samples):
        for i in range(upsample):
            for region in range(regions):
                ts_surrogates[sample][i][region] = sur_func(
                    timeseries[sample][region]
                )
    return ts_surrogates


def calculate_correlation_matrix(timeseries, correlation_type):
    # Placeholder matrices.
    num_subjects, num_regions, _ = timeseries.shape
    corr_matrices = np.empty((num_subjects, num_regions, num_regions))

    for i, ts in enumerate(timeseries):
        if correlation_type.is_symmetric:
            corr_matrices[i] = pd.DataFrame(ts).T.corr(
                method=correlation_type.value
            )
        else:
            corr_matrices[i] = correlation_type.value(ts[:, None], ts)
    return corr_matrices


def xicorr(X: np.array, Y: np.array):
    """
    Correlation metrics from: https://arxiv.org/abs/1910.12327.
    Copied from: https://github.com/czbiohub/xicor/issues/17.
    """
    n = X.size
    # The timeseries do not have duplicates, otherwise use:
    # (X + np.random.rand(len(X))*(10**(-12))).argsort(kind='quicksort')
    xi = np.argsort(X, kind="quicksort")
    Y = Y[xi]
    _, b, c = np.unique(Y, return_counts=True, return_inverse=True)
    r = np.cumsum(c)[b]
    _, b, c = np.unique(-Y, return_counts=True, return_inverse=True)
    l = np.cumsum(c)[b]  # noqa E741 (ambiguous name)
    return 1 - n * np.abs(np.diff(r)).sum() / (2 * (l * (n - l)).sum())


def granger_causality(X: np.array, Y: np.array, lag: int = 1):
    """
    Granger causality chi2 test for a given `lag`.
    """
    # Conform to `statsmodels` package API.
    x = np.vstack([X, Y]).T

    res = sm.tsa.stattools.grangercausalitytests(x, maxlag=[lag], verbose=False)
    return res[lag][0]["ssr_chi2test"][1]


class StratifiedCrossValidation:
    def __init__(
        self,
        targets,
        num_assess_folds,
        num_select_folds,
        random_state,
        single_select_fold: bool = False,
    ):
        self.targets = targets
        self.num_assess_folds = num_assess_folds
        self.num_select_folds = num_select_folds
        self.single_select_fold = single_select_fold

        self._outer_skf = StratifiedKFold(
            n_splits=num_assess_folds, random_state=random_state, shuffle=True
        )
        self._inner_skf = StratifiedKFold(
            n_splits=num_select_folds, random_state=random_state, shuffle=True
        )

        # Iterator over outer CV: assessment of model performance.
        self._outer_cv_iterator = enumerate(
            self._outer_skf.split(np.empty(shape=len(targets)), targets)
        )

    def outer_cross_validation(self):
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
                if self.single_select_fold:
                    self._inner_cv_iterator = iter(())
            except StopIteration:
                # Reset for next experiment.
                self._set_inner_cv_iterator()
                break

            self.train_indices, self.val_indices = (
                self.dev_indices[train_split],
                self.dev_indices[val_split],
            )
            yield i


class DoubleLevelParameterGrid(ParameterGrid):
    """
    Adapted `ParameterGrid` that can handle parameters wrapped in a dictionary.
    """

    def __init__(self, param_grid):
        super().__init__(param_grid)

        # Take all parameters in dictionary and put them outside of it.
        self.expanded_param_grid = []
        self.expanded_dicts = []
        for pg in self.param_grid:
            exp_pg = dict()
            exp_dict = dict()
            for key, value in pg.items():
                if type(value) == dict:
                    exp_dict[key] = value.keys()
                    for k, v in value.items():
                        exp_pg[k] = v
                else:
                    exp_pg[key] = value
            self.expanded_dicts.append(exp_dict)
            self.expanded_param_grid.append(exp_pg)

        self.param_grid, self.orig_param_grid = (
            self.expanded_param_grid,
            self.param_grid,
        )

    def __iter__(self):
        for exp_pg, exp_dict in zip(self.param_grid, self.expanded_dicts):
            # Always sort the keys of a dictionary, for reproducibility.
            items = sorted(exp_pg.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    # Put keys that were originally in a dictionary back inside.
                    for dict_name, dict_keys in exp_dict.items():
                        params[dict_name] = dict()
                        for key in dict_keys:
                            value = params.pop(key)
                            params[dict_name][key] = value

                    yield params

    def __getitem__(self, _):
        return NotImplementedError
