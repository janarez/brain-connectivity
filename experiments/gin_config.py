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
_hyperparameters = {
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
    "eps": [0.0],
    "graph_kwargs": {
        "thresholding_function": [enums.ThresholdingFunction.GROUP_AVERAGE],
        # FIXME: Cannot name file with `str` of `operator` function due to "<>".
        # "thresholding_operator": [operator.ge],
        "threshold_by_absolute_value": [True, False],
        "return_absolute_value": [False],
    },
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

# fixed = {}
# fixed.update(_hyperparameters)
# fixed["graph_kwargs"].update(
#     {
#         "threshold_type": [enums.DataThresholdingType.FIXED_THRESHOLD],
#         "threshold": [0.01, 0.05, 0.1],
#     }
# )

sparsity = {}
sparsity.update(_hyperparameters)
sparsity["graph_kwargs"].update(
    {
        "threshold_type": [enums.DataThresholdingType.SPARSITY],
        "threshold": [10, 20, 30],
    }
)

# knn = {}
# knn.update(_hyperparameters)
# knn["graph_kwargs"].update(
#     {
#         "threshold_type": [enums.DataThresholdingType.KNN],
#         "threshold": [5, 10, 15],
#     }
# )

hyperparameters = [sparsity]
