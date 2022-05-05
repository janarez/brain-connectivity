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
    "node_features": [enums.NodeFeatures.FC_MATRIX_ROW],
    "correlation_type": [
        enums.CorrelationType.PEARSON,
        enums.CorrelationType.SPEARMAN,
        enums.CorrelationType.XI,
    ],
    "batch_size": [16],
    # Model.
    # ======
    "num_hidden_features": [[90, 45]],
    "num_sublayers": [2],
    "graph_kwargs": {
        # Keep all edges.
        "thresholding_function": [enums.ThresholdingFunction.SUBJECT_VALUES],
        "threshold_type": [enums.DataThresholdingType.FIXED_THRESHOLD],
        "threshold": [-100.0],
    },
    # Training.
    # =========
    "criterion": [F.binary_cross_entropy, F.mse_loss, training.cosine_loss],
    "optimizer": [torch.optim.AdamW],
    "optimizer_kwargs": {
        "lr": [0.003, 0.005, 0.0001, 0.0005],
        "weight_decay": [0.0001],
    },
    "epochs": [30],
    "scheduler": [torch.optim.lr_scheduler.LinearLR],
    "scheduler_kwargs": {
        "start_factor": [1],
        "end_factor": [0.01],
        "total_iters": [50],
    },
}
