import os

import torch.nn as nn

from .general_utils import get_logger


class Model(nn.Module):
    def __init__(self, log_folder):
        super(Model, self).__init__()
        self.logger = get_logger("model", os.path.join(log_folder, "model.txt"))

    def forward(self, x):
        raise NotImplementedError

    def plot_fc_matrix(self, epoch, sublayer):
        raise NotImplementedError
