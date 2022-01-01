import torch
import torch.nn.functional as F

from brain_connectivity import enums, training

common_hyperparameters = {
    # Dataset.
    "node_features": [enums.NodeFeatures.FC_MATRIX_ROW],
    "graph_kwargs": [None],
    "upsample_ts": [None],
    "upsample_ts_method": ["iaaft"],
    "correlation_type": [
        enums.CorrelationType.PEARSON,
        # enums.CorrelationType.GRANGER,
        # enums.CorrelationType.SPEARMAN,
        # enums.CorrelationType.XI,
    ],
    "batch_size": [4, 8, 16, 32],
    # Model.
    "num_hidden_features": [4, 8, 16, 32],
    "num_sublayers": [1],
    "dropout": [0.1, 0.3, 0.5],
    # Training.
    "criterion": [
        F.binary_cross_entropy
    ],  # , F.mse_loss, training.cosine_loss],
    "optimizer": [torch.optim.Adam],
    "optimizer_kwargs": {
        "lr": [0.001],  # , 0.005, 0.0001, 0.0005],
        "weight_decay": [0.0001],
        # "momentum": [0.3],
    },
    "epochs": [30],
    "scheduler": [torch.optim.lr_scheduler.ReduceLROnPlateau],
    "scheduler_kwargs": {"factor": [0.5], "patience": [1]},
}

graph_hyperparameters_fixed = {}
# First put in common to override `graph_kwargs` properly.
graph_hyperparameters_fixed.update(common_hyperparameters)
graph_hyperparameters_fixed.update(
    {
        # How to create FC matrix.
        "graph_kwargs": {
            "thresholding_function": [
                enums.ThresholdingFunction.GROUP_AVERAGE,
                enums.ThresholdingFunction.SUBJECT_VALUES,
            ],
            "threshold_type": [enums.DataThresholdingType.FIXED_THRESHOLD],
            "threshold": [0.01, 0.05, 0.1],
            # FIXME: Cannot name file with `str` of `operator` function due to "<>".
            # "thresholding_operator": [operator.ge],
            "threshold_by_absolute_value": [True, False],
            "return_absolute_value": [False],
        },
        "eps": [0.0],
    }
)

graph_hyperparameters_knn = {}
graph_hyperparameters_knn.update(graph_hyperparameters_fixed)
graph_hyperparameters_knn["graph_kwargs"]["threshold_type"] = [
    enums.DataThresholdingType.KNN
]
graph_hyperparameters_knn["graph_kwargs"]["threshold"] = [5, 10, 20]
graph_hyperparameters = [graph_hyperparameters_fixed, graph_hyperparameters_knn]

dense_hyperparameters = {
    "mode": [enums.ConnectivityMode.SINGLE],
    "num_nodes": [90],
    "readout": ["add"],  # "mean", "max"],
    "emb_dropout": [0.0],
    "emb_residual": ["add"],  # [None, "add"],
    "emb_init_weights": ["constant"],
    "emb_val": [0.0],
    "emb_std": [0.01],
}
dense_hyperparameters.update(common_hyperparameters)

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
