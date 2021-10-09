from enum import Enum, auto
from typing import List, Union
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


class ConnectivityEmbedding(nn.Module):
    """
    Learns connectivity between nodes. For each node a weighted combination of all the nodes is learned.

    Input: [batch_size, num_nodes, num_features]
    Output: [batch_size, num_nodes, num_features]
    """
    def __init__(self, size, dropout: 0.0):
        super(ConnectivityEmbedding, self).__init__()
        # Initialize with fully connected graph.
        self.fc_matrix = nn.Parameter(torch.ones(size, size), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)


    def toggle_gradients(self, requires_grad):
        self.fc_matrix.requires_grad = requires_grad


    def forward(self, x):
        # There is no non-linearity since we are just combining nodes.
        return self.dropout(torch.matmul(self.fc_matrix, x))


class ConnectivityMLP(nn.Module):
    """
    Runs node features through MLP.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, num_nodes, num_out_features]
    """
    def __init__(self, size_in, size_out, dropout):
        super(ConnectivityMLP, self).__init__()
        self.fc = nn.Linear(size_in, size_out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return F.relu(self.dropout(self.fc(x)))


class ConnectivityMode(Enum):
    """
    Determines how is connectivity matrix obtained.

    FIXED: Use handmade FC matrix.
    START: Learn FC matrix only on raw input features.
    SINGLE: Learn FC matrix on raw input features as well as all subsequent feature mapping layers.
    MULTIPLE: Learn new FC matrix before every feature mapping layer.
    """
    FIXED = auto()
    START = auto()
    SINGLE = auto()
    MULTIPLE = auto()


class ConnectivitySublayer(nn.Module):
    """
    Combines neighborhood connectivity with MLP transformation.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, num_nodes, num_out_features]
    """
    def __init__(self, sublayer_id: int, size_in: int, size_out: int, dropout: float, mode: ConnectivityMode, **mode_kwargs):
        super(ConnectivitySublayer, self).__init__()

        self.id = sublayer_id
        # Create new FC matrix for this sublayer.
        if mode == ConnectivityMode.MULTIPLE:
            self.fc_matrix = ConnectivityEmbedding(size_in, dropout=mode_kwargs['fc_dropout'])
        # Used passed in FC matrix.
        else:
            self.fc_matrix = mode_kwargs['fc_matrix']
            # Switch of gradients for subsequent layers in start mode.
            if mode == mode.START and sublayer_id > 0:
                self.fc_matrix.toggle_gradients(requires_grad=False)

        # Feature mapping layer.
        self.mlp = ConnectivityMLP(size_in, size_out, dropout)

    def forward(self, x):
        # Aggregate feature vectors based on connectivity neighborhood.
        x = self.fc_matrix(x)
        # Map features.
        x = self.mlp(x)
        return x


class ConnectivityDenseNet(nn.Module):
    """
    Emulates Graph isomorphism network using a fully connected alternative.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, 2]
    """
    def __init__(
        self,
        num_nodes: int,
        mode: ConnectivityMode,
        size_in: int,
        num_hidden_features: Union[int, List[int]],
        dropout: float = 0.5,
        connectivity_dropout: float = 0.0,
        num_sublayers: int = 3,
        readout: str = 'add',
        **mode_kwargs
    ):
        super(ConnectivityDenseNet, self).__init__()

        self.mode = mode
        self.fc_matrix = None
        # Set passed in FC matrix.
        if mode == ConnectivityMode.FIXED:
            self.fc_matrix = lambda x: torch.matmul(mode_kwargs['fc_matrix'], x)
        # Create single FC matrix that will be learned only at the beggining.
        # or
        # Create single FC matrix that will be learned throughout.
        elif (mode == ConnectivityMode.START) or (mode == ConnectivityMode.SINGLE):
            self.fc_matrix = ConnectivityEmbedding(num_nodes, dropout=connectivity_dropout)
        # Else `ConnectivityMode.MULTIPLE`, let each sublayer create its own FC matrix.
        self.mode_kwargs = {
            'fc_matrix': self.fc_matrix,
            'fc_dropout': connectivity_dropout
        }

        # Prepare feature mapping dimensions.
        if type(num_hidden_features) is int:
            num_out_features = np.repeat(num_hidden_features, num_sublayers)
        num_in_features = copy.copy(num_out_features)
        num_in_features[0] = size_in

        # Create model stacked from sublayers: connectivity + feature mapping.
        self.sublayers = nn.ModuleList([
            ConnectivitySublayer(
                i, size_in, size_out, dropout=dropout, mode=mode, **self.mode_kwargs
            ) for i, (size_in, size_out) in enumerate(zip(num_in_features, num_out_features))
        ])

        # Classification head.
        self.readout = readout
        self.fc = nn.Linear(num_out_features[-1], 2)


    def forward(self, data):
        x = data.x

        # Run sample through model.
        for sublayer in self.sublayers:
            x = sublayer(x)

        # Turn on gradients for start mode back on.
        if self.mode == ConnectivityMode.START:
            self.fc_matrix.toggle_gradients(requires_grad=True)

        # Binary classification head.
        # Readout across nodes.
        if self.readout == 'add':
            x = torch.sum(x, dim=1)
        elif self.readout == 'mean':
            x = torch.mean(x, dim=1)
        if self.readout == 'max':
            x = torch.max(x, dim=1)

        # Return binary logits.
        return self.fc(x)
