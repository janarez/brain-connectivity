import torch
import torch.nn.functional as F
from brain_connectivity import enums, training

# Always fixed parameters.
# ==============================================================================
model_params = {"size_in": 4005}  # 8100 for flattened,  4005 for triangular.
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
    "correlation_type": [enums.CorrelationType.SPEARMAN],
    "batch_size": [16, 32],
    # Model.
    # ======
    "num_hidden_features": [10, 20],
    "num_sublayers": [1],
    "dropout": [0.3],
    # Training.
    # =========
    "criterion": [F.mse_loss, training.cosine_loss, F.binary_cross_entropy],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.1, 0.5],
        "weight_decay": [0.0001],
    },
    "epochs": [20],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.01],
        "total_iters": [20],
    },
}
