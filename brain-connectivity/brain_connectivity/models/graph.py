from typing import List, Union

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import BatchNorm, GATConv, GINConv

from .model import Model, mlp_dimensions


class GinMLP(nn.Module):
    def __init__(self, size_in, size_out, dropout):
        super().__init__()

        self.fc = nn.Linear(size_in, size_out)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.dropout(self.activation(self.fc(x)))


class GIN(Model):

    hyperparameters = ["num_hidden_features", "num_sublayers", "dropout", "eps"]

    def __init__(
        self,
        size_in: int,
        num_hidden_features: Union[List[int], int] = 90,
        dropout: float = 0.5,
        num_sublayers: int = 3,
        eps: float = 0.0,
    ):
        super().__init__()

        num_in_features, num_out_features = mlp_dimensions(
            size_in, num_hidden_features, num_sublayers
        )

        self.convs = nn.ModuleList(
            [
                GINConv(nn=GinMLP(size_in, size_out, dropout=dropout), eps=eps)
                for size_in, size_out in zip(num_in_features, num_out_features)
            ]
        )
        self.bns = nn.ModuleList(
            [BatchNorm(in_channels=size) for size in num_out_features]
        )

        self.fc = nn.Linear(num_out_features[-1], 1)
        # self.fc = nn.Linear(sum(num_out_features), 1)
        self.activation = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing + readout.
        # out = []
        for bn, conv in zip(self.bns, self.convs):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            # out.append(torch_geometric.nn.global_mean_pool(x, batch))
        # x = torch.cat(out, 1)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        # Classification FC.
        x = torch.sigmoid(self.fc(x))
        return x


class GAT(Model):

    hyperparameters = ["num_hidden_features", "num_sublayers"]

    def __init__(
        self,
        size_in: int,
        num_hidden_features: Union[List[int], int] = 90,
        num_sublayers: int = 3,
    ):
        super().__init__()

        num_in_features, num_out_features = mlp_dimensions(
            size_in, num_hidden_features, num_sublayers
        )

        self.convs = nn.ModuleList(
            [
                GATConv(in_channels=int(size_in), out_channels=size_out)
                for size_in, size_out in zip(num_in_features, num_out_features)
            ]
        )
        self.bns = nn.ModuleList(
            [BatchNorm(in_channels=size) for size in num_out_features]
        )

        self.activation = nn.ReLU()
        self.fc = nn.Linear(num_out_features[-1], 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing.
        for bn, conv in zip(self.bns, self.convs):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
        # Readout.
        x = torch_geometric.nn.global_mean_pool(x, batch)
        # Classification FC.
        x = torch.sigmoid(self.fc(x))

        return x
