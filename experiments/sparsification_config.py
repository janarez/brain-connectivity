import torch
import torch.nn.functional as F
from brain_connectivity import enums

# Always fixed parameters.
# ==============================================================================
model_params = {"size_in": 90}
training_params = {
    # Plotting.
    "fc_matrix_plot_frequency": None,
    "fc_matrix_plot_sublayer": 0,
}


# Hyperparameters.
# ==============================================================================
knn = {
    # Dataset.
    # ========
    "node_features": [
        enums.NodeFeatures.FC_MATRIX_ROW,
        enums.NodeFeatures.ONE_HOT_REGION,
    ],
    "correlation_type": [enums.CorrelationType.PEARSON],
    "batch_size": [32],
    # Model.
    # ======
    "num_hidden_features": [45],
    "num_sublayers": [2],
    "dropout": [0.5],
    "eps": [0.0],
    "graph_kwargs": {
        "thresholding_function": [
            enums.ThresholdingFunction.SUBJECT_VALUES,
            enums.ThresholdingFunction.RANDOM,
        ],
        "threshold_by_absolute_value": [False],
        "return_absolute_value": [False],
        "threshold_type": [enums.DataThresholdingType.KNN],
        "threshold": [3, 5, 10, 15, 20, 30, 40],
    },
    # Training.
    # =========
    "criterion": [F.mse_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {"lr": [0.003], "weight_decay": [0.0001]},
    "epochs": [100],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.01],
        "total_iters": [50],
    },
}

sparsity = {
    # Dataset.
    # ========
    "node_features": [
        enums.NodeFeatures.FC_MATRIX_ROW,
        enums.NodeFeatures.ONE_HOT_REGION,
        enums.NodeFeatures.ONE,
    ],
    "correlation_type": [enums.CorrelationType.PEARSON],
    "batch_size": [32],
    # Model.
    # ======
    "num_hidden_features": [45],
    "num_sublayers": [2],
    "dropout": [0.5],
    "eps": [0.0],
    "graph_kwargs": {
        "thresholding_function": [
            enums.ThresholdingFunction.SUBJECT_VALUES,
            enums.ThresholdingFunction.RANDOM,
        ],
        "threshold_by_absolute_value": [False],
        "return_absolute_value": [False],
        "threshold_type": [enums.DataThresholdingType.SPARSITY],
        "threshold": [5, 10, 15, 20, 25, 30, 40],
    },
    # Training.
    # =========
    "criterion": [F.mse_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {"lr": [0.003], "weight_decay": [0.0001]},
    "epochs": [100],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.01],
        "total_iters": [50],
    },
}


hyperparameters = [knn, sparsity]
