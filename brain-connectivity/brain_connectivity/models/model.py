import inspect
import io
import os
from contextlib import redirect_stdout

import numpy as np
import torch.nn as nn
from torchinfo import summary

from ..general_utils import close_logger, get_logger


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def log(cls, log_folder, kwargs):
        logger = get_logger("model", os.path.join(log_folder, "model.txt"))
        signature = inspect.signature(cls.__init__)

        for key, value in signature.parameters.items():
            v = kwargs.get(key, value.default)
            logger.debug(f"{' '.join(key.capitalize().split('_'))}: {v}")

        # Necessary, because `summary` explicitly calls `print` if verbose is 2.
        with redirect_stdout(io.StringIO()) as f:
            summary(cls(**kwargs), verbose=2)
        logger.debug(f"Architecture:\n{f.getvalue()}")
        close_logger("model")

    def forward(self, x):
        raise NotImplementedError

    def plot_fc_matrix(self, epoch, sublayer):
        raise NotImplementedError

    def _mlp_dimensions(self, size_in, num_hidden_features, num_sublayers):
        if isinstance(num_hidden_features, int):
            num_out_features = np.repeat(num_hidden_features, num_sublayers)
        else:
            num_out_features = num_hidden_features
        num_in_features = np.hstack([[size_in], num_out_features])
        return num_in_features, num_out_features
