import torch
import torch.nn.functional as F
from brain_connectivity import enums, training

# Always fixed parameters.
# ==============================================================================
model_params = {"size_in": 90}
training_params = {
    # Training regime.
    "validation_frequency": 1,
    # Plotting.
    "fc_matrix_plot_frequency": None,
    "fc_matrix_plot_sublayer": 0,
}


# Hyperparameters.
# ==============================================================================
hyperparameters = {
    # Dataset.
    # ========
    "node_features": [enums.NodeFeatures.FC_MATRIX_ROW],
    "upsample_ts": [None],
    "upsample_ts_method": ["iaaft"],
    "correlation_type": [
        enums.CorrelationType.PEARSON,
        # enums.CorrelationType.GRANGER,
        # enums.CorrelationType.SPEARMAN,
        # enums.CorrelationType.XI,
    ],
    "batch_size": [2],  # , 4, 8],
    # Model.
    # ======
    "num_hidden_features": [2, 4],
    "num_sublayers": [1],
    "dropout": [0.5],
    "mode": [enums.ConnectivityMode.SINGLE],
    "num_nodes": [90],
    "readout": ["add"],  # "mean", "max"],
    "emb_dropout": [0.0],
    "emb_residual": ["add"],  # [None, "add"],
    "emb_init_weights": ["constant"],
    "emb_val": [0.0],
    "emb_std": [0.01],
    "graph_kwargs": [None],
    # Training.
    # =========
    "criterion": [
        F.binary_cross_entropy
    ],  # , F.mse_loss, training.cosine_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.001],  # , 0.005, 0.0001, 0.0005],
        "weight_decay": [0.0001],
        # "momentum": [0.3],
    },
    "epochs": [30],
    "scheduler": [torch.optim.lr_scheduler.ReduceLROnPlateau],
    "scheduler_kwargs": {"factor": [0.5], "patience": [1]},
}
