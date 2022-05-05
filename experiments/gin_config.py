import torch
import torch.nn.functional as F
from brain_connectivity import enums, training

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
hyperparameters = {
    # Dataset.
    # ========
    "node_features": [
        enums.NodeFeatures.ONE_HOT_REGION,
    ],
    "correlation_type": [enums.CorrelationType.PEARSON],
    "batch_size": [32],
    "upsample_ts_method": ["iaaft"],
    "upsample_ts": [1],
    # Model.
    # ======
    "num_hidden_features": [45],
    "num_sublayers": [2],
    "dropout": [0.5],
    "eps": [0.0],
    "graph_kwargs": {
        "thresholding_function": [enums.ThresholdingFunction.SUBJECT_VALUES],
        "threshold_by_absolute_value": [False],
        "return_absolute_value": [False],
        "threshold_type": [enums.DataThresholdingType.KNN],
        "threshold": [10],
    },
    # Training.
    # =========
    "criterion": [F.mse_loss, training.cosine_loss, F.binary_cross_entropy],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {"lr": [0.003], "weight_decay": [0.0001]},
    "epochs": [50],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.01],
        "total_iters": [50],
    },
}
