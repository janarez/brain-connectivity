import torch
import torch.nn.functional as F
from brain_connectivity import enums, training

# Always fixed parameters.
# ==============================================================================
model_params = {
    "size_in": 90,
    "num_nodes": 90,
    "mode": enums.ConnectivityMode.SINGLE,
}
training_params = {
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
        enums.CorrelationType.PEARSON,
        enums.CorrelationType.SPEARMAN,
        enums.CorrelationType.XI,
    ],
    "batch_size": [2, 4, 8],
    # Model.
    # ======
    "num_hidden_features": [10, 20, 45],
    "num_sublayers": [2],
    "dropout": [0.5],
    "readout": ["mean", "add", "max"],
    "emb_dropout": [0.1],
    "emb_residual": [None],
    "emb_init_weights": [
        "constant",
        "normal",
    ],
    "emb_val": [0.0],
    "emb_std": [0.0001],
    # Training.
    # =========
    "criterion": [F.mse_loss, training.cosine_loss, F.binary_cross_entropy],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.1, 0.01],
        "weight_decay": [0.0005],
    },
    "epochs": [50],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.0001],
        "total_iters": [50],
    },
}
