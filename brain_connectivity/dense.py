from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .enums import ConnectivityMode
from .model import Model
from .visualizations import plot_fc_matrix


class ConnectivityEmbedding(nn.Module):
    """
    Learns connectivity between nodes. For each node a weighted combination of all the nodes is learned.

    Input: [batch_size, num_nodes, num_features]
    Output: [batch_size, num_nodes, num_features]
    """

    def __init__(self, size, dropout, residual, init_weights, val, std):
        super(ConnectivityEmbedding, self).__init__()
        # Initialize with fully connected graph.
        self.fc_matrix = nn.Parameter(
            torch.empty(size, size), requires_grad=True
        )

        if init_weights == "constant":
            nn.init.constant_(self.fc_matrix, val=val)
        elif init_weights == "normal":
            nn.init.normal_(self.fc_matrix, mean=val, std=std)
        elif init_weights == "uniform":
            nn.init.uniform_(self.fc_matrix, a=-val, b=val)
        elif init_weights == "sparse":
            nn.init.sparse_(self.fc_matrix, sparsity=val, std=std)
        else:
            raise ValueError(
                f"Found {init_weights}, expected one of: 'constant', 'normal', 'uniform' or 'sparse' for param `init_weights`"
            )

        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual

    def toggle_gradients(self, requires_grad):
        self.fc_matrix.requires_grad = requires_grad

    def forward(self, x):
        # There is no non-linearity since we are just combining nodes.
        x_neighborhoods = self.dropout(torch.matmul(self.fc_matrix, x))

        # Combine with original input for residual connection.
        if self.residual == "add":
            x = x + x_neighborhoods
        elif self.residual == "mean":
            x = torch.mean(torch.stack([x, x_neighborhoods]), dim=0)
        elif self.residual == "max":
            x = torch.max(torch.stack([x, x_neighborhoods]), dim=0).values

        return x


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


class ConnectivitySublayer(nn.Module):
    """
    Combines neighborhood connectivity with MLP transformation.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, num_nodes, num_out_features]
    """

    def __init__(
        self,
        sublayer_id: int,
        size_in: int,
        size_out: int,
        size_emb: int,
        dropout: float,
        mode: ConnectivityMode,
        mode_kwargs: dict,
        emb_matrix: Optional[ConnectivityEmbedding] = None,
    ):
        super(ConnectivitySublayer, self).__init__()

        self.id = sublayer_id
        # Create new FC matrix for this sublayer.
        if mode == ConnectivityMode.MULTIPLE:
            self.fc_matrix = ConnectivityEmbedding(size_emb, **mode_kwargs)
        # Used passed in FC matrix.
        else:
            self.fc_matrix = emb_matrix
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


class ConnectivityDenseNet(Model):
    """
    Emulates Graph isomorphism network using a fully connected alternative.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, 2]
    """

    hyperparameters = [
        "num_hidden_features",
        "num_sublayers",
        "dropout",
        "mode",
        "num_nodes",
        "readout",
        "emb_dropout",
        "emb_residual",
        "emb_init_weights",
        "emb_val",
        "emb_std",
    ]

    def __init__(
        self,
        num_nodes: int,
        mode: ConnectivityMode,
        size_in: int,
        num_hidden_features: Union[int, List[int]],
        dropout: float = 0.5,
        emb_dropout: float = 0.0,
        emb_residual: Optional[str] = "mean",
        emb_init_weights: str = "constant",
        emb_val: float = 0.0,
        emb_std: float = 0.01,
        num_sublayers: int = 3,
        readout: str = "add",
        **mode_kwargs,
    ):
        super(ConnectivityDenseNet, self).__init__()

        self.mode = mode
        self.fc_matrix = None
        emb_kwargs = {
            "dropout": emb_dropout,
            "residual": emb_residual,
            "init_weights": emb_init_weights,
            "val": emb_val,
            "std": emb_std,
        }
        # Set passed in FC matrix.
        if mode == ConnectivityMode.FIXED:
            self.fc_matrix = lambda x: torch.matmul(mode_kwargs["fc_matrix"], x)
        # Create single FC matrix that will be learned only at the beggining.
        # or
        # Create single FC matrix that will be learned throughout.
        elif (mode == ConnectivityMode.START) or (
            mode == ConnectivityMode.SINGLE
        ):
            self.fc_matrix = ConnectivityEmbedding(num_nodes, **emb_kwargs)
        # Else `ConnectivityMode.MULTIPLE`, let each sublayer create its own FC matrix.

        # Prepare feature mapping dimensions.
        if type(num_hidden_features) is int:
            num_out_features = np.repeat(num_hidden_features, num_sublayers)
        else:
            num_out_features = num_hidden_features
        num_in_features = np.hstack([[size_in], num_out_features])

        # Create model stacked from sublayers: connectivity + feature mapping.
        self.sublayers = nn.ModuleList(
            [
                ConnectivitySublayer(
                    i,
                    size_in,
                    size_out,
                    size_emb=num_nodes,
                    dropout=dropout,
                    mode=mode,
                    mode_kwargs=emb_kwargs,
                    emb_matrix=self.fc_matrix,
                )
                for i, (size_in, size_out) in enumerate(
                    zip(num_in_features, num_out_features)
                )
            ]
        )

        # Classification head.
        self.readout = readout
        self.fc = nn.Linear(num_out_features[-1], 1)

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
        if self.readout == "add":
            x = torch.sum(x, dim=1)
        elif self.readout == "mean":
            x = torch.mean(x, dim=1)
        elif self.readout == "max":
            x = torch.max(x, dim=1).values

        # Return probabilities.
        return torch.sigmoid(self.fc(x))

    def plot_fc_matrix(self, epoch, sublayer=0):
        # TODO: Adapt for any connectivity mode.
        fc_matrix = self.sublayers[sublayer].fc_matrix.fc_matrix
        # FIXME: Is it okey to call `detach`?
        numpy_fc_matrix = fc_matrix.cpu().detach().numpy()
        print(
            fc_matrix.sum(),
            numpy_fc_matrix.sum(),
            numpy_fc_matrix.mean(),
            numpy_fc_matrix.std(),
        )
        plot_fc_matrix(matrix=numpy_fc_matrix, epoch=epoch)


class DenseNet(Model):
    """
    Emulates Graph isomorphism network using a fully connected alternative.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, 2]
    """

    hyperparameters = [
        "num_hidden_features",
        "num_sublayers",
        "dropout",
    ]

    def __init__(
        self,
        size_in: int,
        num_hidden_features: Union[int, List[int]],
        dropout: float = 0.5,
        num_sublayers: int = 3,
    ):
        super().__init__()

        # Prepare feature mapping dimensions.
        # TODO: Refactor, it's used in all models.
        if type(num_hidden_features) is int:
            num_out_features = np.repeat(num_hidden_features, num_sublayers)
        else:
            num_out_features = num_hidden_features
        num_in_features = np.hstack([[size_in ** 2], num_out_features])

        # Create model stacked from linear sublayers.
        self.sublayers = nn.ModuleList(
            [
                nn.Linear(size_in, size_out)
                for size_in, size_out in zip(num_in_features, num_out_features)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)

        # Classification head.
        self.fc = nn.Linear(num_out_features[-1], 1)

    def forward(self, data):
        x = data.x

        # Run sample through model.
        for sublayer in self.sublayers:
            x = sublayer(x)
            x = self.dropout(x)

        # Return probabilities.
        return torch.sigmoid(self.fc(x))

    def plot_fc_matrix(self, epoch, sublayer=0):
        # TODO: Adapt for any connectivity mode.
        raise NotImplementedError
