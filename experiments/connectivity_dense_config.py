import torch
import torch.nn.functional as F
from brain_connectivity import enums  # training

# Always fixed parameters.
# ==============================================================================
model_params = {
    "size_in": 90,
    "num_nodes": 90,
    "mode": enums.ConnectivityMode.SINGLE,
}
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
    "correlation_type": [
        # enums.CorrelationType.PEARSON,
        # enums.CorrelationType.GRANGER,
        enums.CorrelationType.SPEARMAN,
        # enums.CorrelationType.XI,
    ],
    "batch_size": [2],
    # Model.
    # ======
    "num_hidden_features": [45, 30],
    "num_sublayers": [2],
    "dropout": [0.3],
    "readout": ["max", "add"],
    "emb_dropout": [0.0, 0.1],
    "emb_residual": [None],  # [None, "add"],
    "emb_init_weights": ["normal", "constant"],
    "emb_val": [0.0],  # [0.0001],
    "emb_std": [0.0001],
    # Training.
    # =========
    "criterion": [F.mse_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.005],
        "weight_decay": [0.0001],
        # "momentum": [0.3],
    },
    "epochs": [50],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.005],
        "total_iters": [50],
    },
}
