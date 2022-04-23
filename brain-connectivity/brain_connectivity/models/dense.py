from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..enums import ConnectivityMode
from ..visualizations import fc_matrix_heatmap
from .model import Model


class ConnectivityEmbedding(nn.Module):
    """
    Learns connectivity between nodes. For each node a weighted combination of all the nodes is learned.

    Input: [batch_size, num_nodes, num_features]
    Output: [batch_size, num_nodes, num_features]
    """

    def __init__(self, size, dropout, residual, init_weights, val, std=None):
        super().__init__()
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
        self.init_weights = init_weights

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
        else:
            x = x_neighborhoods

        return x


class ConnectivityMLP(nn.Module):
    """
    Runs node features through MLP.

    Input: [batch_size, num_nodes, num_in_features]
    Output: [batch_size, num_nodes, num_out_features]
    """

    def __init__(self, size_in, size_out, dropout):
        super().__init__()
        self.fc = nn.Linear(size_in, size_out)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(self.activation(x))
        return x


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
        super().__init__()

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
        super().__init__()

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
        num_in_features, num_out_features = self._mlp_dimensions(
            size_in, num_hidden_features, num_sublayers
        )

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

    def plot_fc_matrix(self, epoch, sublayer, path):
        # TODO: Adapt for any connectivity mode.
        fc_matrix_emb = self.sublayers[sublayer].fc_matrix
        numpy_fc_matrix = fc_matrix_emb.fc_matrix.cpu().detach().numpy()
        title = f"FC matrix at {epoch} epochs, init: {fc_matrix_emb.init_weights}, residual: {fc_matrix_emb.residual}"
        fc_matrix_heatmap(
            matrix=numpy_fc_matrix, epoch=epoch, path=path, title=title
        )


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

        self.size_in = size_in
        num_in_features, num_out_features = self._mlp_dimensions(
            size_in, num_hidden_features, num_sublayers
        )

        # Create model stacked from linear sublayers.
        self.sublayers = nn.ModuleList(
            [
                nn.Linear(size_in, size_out)
                for size_in, size_out in zip(num_in_features, num_out_features)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(size_out) for size_out in num_out_features]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        # Classification head.
        self.fc = nn.Linear(num_out_features[-1], 1)

    def forward(self, data):
        x = data.x

        # Run sample through model.
        for norm, sublayer in zip(self.norms, self.sublayers):
            x = norm(sublayer(x))
            x = self.dropout(self.activation(x))

        # Return probabilities.
        return torch.sigmoid(self.fc(x))

    def plot_fc_matrix(self, epoch, sublayer, path):
        dense = self.sublayers[sublayer].weight
        assert (
            torch.numel(dense) == self.size_in
        ), f"Can only deflattened `size_in`**2 matrices, not {torch.numel(dense)}"
        num_nodes = int(self.size_in ** 0.5)
        numpy_fc_matrix = (
            dense.reshape(num_nodes, num_nodes).cpu().detach().numpy()
        )
        fc_matrix_heatmap(matrix=numpy_fc_matrix, epoch=epoch, path=path)
