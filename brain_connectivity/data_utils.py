import torch


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def zeroth_axis_sample(matrix, i):
    return torch.from_numpy(matrix[i]).to(torch.float32)


def identity_matrix(size, _):
    "Keep last parameter for index in unified API."
    return torch.diag(torch.ones(size))
