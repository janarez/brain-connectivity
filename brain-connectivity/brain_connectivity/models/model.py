"""
Base model class and model utilities.
"""

import inspect
import io
import os
from contextlib import redirect_stdout

import numpy as np
import torch
from torchinfo import summary

from ..general_utils import close_logger, get_logger


class Model(torch.nn.Module):
    """
    Base model class.
    """

    def __init__(self, binary_cls):
        super().__init__()
        self.binary_cls = binary_cls

    @classmethod
    def log(cls, log_folder, kwargs):
        """
        Logs model architecture and parameters.
        """
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

    def output_activation(self, x):
        return x if not self.binary_cls else torch.sigmoid(x)


def mlp_dimensions(size_in, num_hidden_features, num_sublayers):
    """
    Helper for deriving hidden dense layers dimensions.
    """
    if isinstance(num_hidden_features, int):
        num_out_features = np.repeat(num_hidden_features, num_sublayers)
    else:
        num_out_features = num_hidden_features
    num_in_features = np.hstack([[size_in], num_out_features])
    return num_in_features, num_out_features
