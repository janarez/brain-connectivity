import inspect
import os

import torch.nn as nn
from torchinfo import summary

from .general_utils import close_logger, get_logger


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    @classmethod
    def log(cls, log_folder, kwargs):
        logger = get_logger("model", os.path.join(log_folder, "model.txt"))
        signature = inspect.signature(cls.__init__)

        for key, value in signature.parameters.items():
            v = kwargs.get(key, value.default)
            logger.debug(f"{' '.join(key.capitalize().split('_'))}: {v}")

        logger.debug(summary(cls.__init__(**kwargs), verbose=2))
        close_logger("model")

    def forward(self, x):
        raise NotImplementedError

    def plot_fc_matrix(self, epoch, sublayer):
        raise NotImplementedError
