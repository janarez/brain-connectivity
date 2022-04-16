import torch
import torch.nn.functional as F
from brain_connectivity import enums, training

# Always fixed parameters.
# ==============================================================================
model_params = {"size_in": 4005}  # 8100 for flattened,  4005 for triangular.
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
        enums.CorrelationType.PEARSON,
        enums.CorrelationType.SPEARMAN,
        enums.CorrelationType.XI,
    ],
    "batch_size": [2, 4, 8],  # , 4, 8],
    # Model.
    # ======
    "num_hidden_features": [2],
    "num_sublayers": [1],  # , 2, 3],
    "dropout": [0.3],
    # Training.
    # =========
    "criterion": [
        training.cosine_loss
    ],  # F.binary_cross_entropy, F.mse_loss, training.cosine_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.001],
        "weight_decay": [0.0001],
        # "momentum": [0.3],
    },
    "epochs": [50],
    # "scheduler": [torch.optim.lr_scheduler.LinearLR],
    # "scheduler_kwargs": {
    #     "start_factor": [1],
    #     "end_factor": [0.001],
    #     "total_iters": [25, 50],
    # },
    "scheduler": [torch.optim.lr_scheduler.ReduceLROnPlateau],
    "scheduler_kwargs": {"factor": [0.5], "patience": [1]},
}
