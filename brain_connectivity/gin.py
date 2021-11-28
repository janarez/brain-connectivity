from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

from .model import Model


class GinMLP(nn.Module):
    def __init__(self, size_in, size_out, dropout):
        super(GinMLP, self).__init__()

        self.fc = nn.Linear(size_in, size_out)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dropout(self.fc(x)))


class GIN(Model):
    def __init__(
        self,
        log_folder: str,
        size_in: int,
        num_hidden_features: Union[List[int], int] = 90,
        dropout: float = 0.5,
        num_sublayers: int = 3,
        eps: float = 0.0,
    ):
        super(GIN, self).__init__(log_folder)

        # Prepare feature mapping dimensions.
        if type(num_hidden_features) is int:
            num_out_features = np.repeat(num_hidden_features, num_sublayers)
        else:
            num_out_features = num_hidden_features
        num_in_features = np.hstack([[size_in], num_out_features])

        self.convs = torch.nn.ModuleList(
            [
                GINConv(nn=GinMLP(size_in, size_out, dropout=dropout), eps=eps)
                for size_in, size_out in zip(num_in_features, num_out_features)
            ]
        )
        self.fc = nn.Linear(num_out_features[-1], 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing.
        for conv in self.convs:
            x = conv(x, edge_index)
        # Readout.
        x = global_add_pool(x, batch)
        # Classification FC.
        x = self.fc(x)

        return x
