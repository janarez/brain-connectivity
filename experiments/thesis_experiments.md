# Configurations for all schizophrenia experiments reported in "Utilization of brain connectivity in classification and regression tasks in brain data" diploma thesis.


## Schizophrenia discrimination

### Standard ML models

- Config: `ml_config.py`
- Script: `run_ml_experiment.py`

### Deep learning models

For all deep learning models, the hyperparameter space was first explored using the `--single_select_fold` option of `run_experiment.py`.

#### Dense model

- Config: `dense_config.py`
- Script: `run_experiment.py` with `triangular_dense` as `model_type`

Results are reported for this configuration:

`[{'node_features': [<NodeFeatures.FC_MATRIX_ROW: 1>], 'correlation_type': [<CorrelationType.XI: functools.partial(<function xicorr at 0x000002142869AAF8>)>], 'batch_size': [4, 8, 16], 'num_hidden_features': [2, 4], 'num_sublayers': [1], 'dropout': [0.3], 'criterion': [<function cosine_loss at 0x000002142FEFF708>], 'optimizer': [<class 'torch.optim.adamw.AdamW'>], 'optimizer_kwargs': {'lr': [0.01], 'weight_decay': [0.0001]}, 'epochs': [50], 'scheduler': [<class 'torch.optim.lr_scheduler.LinearLR'>], 'scheduler_kwargs': {'start_factor': [1], 'end_factor': [0.01], 'total_iters': [50]}}]`

#### Proposed model

- Config: `connectivity_dense_config.py`
- Script: `run_experiment.py` with `connectivity_dense` as `model_type`

Results are reported for this configuration:

`[{'node_features': [<NodeFeatures.FC_MATRIX_ROW: 1>], 'correlation_type': [<CorrelationType.XI: functools.partial(<function xicorr at 0x000002142869AAF8>)>], 'batch_size': [4, 8, 16], 'num_hidden_features': [2, 4], 'num_sublayers': [1], 'dropout': [0.3], 'criterion': [<function cosine_loss at 0x000002142FEFF708>], 'optimizer': [<class 'torch.optim.adamw.AdamW'>], 'optimizer_kwargs': {'lr': [0.01], 'weight_decay': [0.0001]}, 'epochs': [50], 'scheduler': [<class 'torch.optim.lr_scheduler.LinearLR'>], 'scheduler_kwargs': {'start_factor': [1], 'end_factor': [0.01], 'total_iters': [50]}}]`

#### GAT model

- Config: `gat_config.py`
- Script: `run_experiment.py` with `gat` as `model_type`

Results are reported for this configuration:

`[{'node_features': [<NodeFeatures.FC_MATRIX_ROW: 1>], 'correlation_type': [<CorrelationType.PEARSON: 'pearson'>, <CorrelationType.XI: <numpy.vectorize object at 0x00000251C3D814C8>>], 'batch_size': [16], 'num_hidden_features': [45], 'num_sublayers': [1, 2], 'graph_kwargs': {'thresholding_function': [<ThresholdingFunction.SUBJECT_VALUES: 2>], 'threshold_type': [<DataThresholdingType.FIXED_THRESHOLD: 2>], 'threshold': [-100.0]}, 'criterion': [<function mse_loss at 0x00000251B0253F78>], 'optimizer': [<class 'torch.optim.adamw.AdamW'>], 'optimizer_kwargs': {'lr': [0.003], 'weight_decay': [0.0001]}, 'epochs': [30], 'scheduler': [<class 'torch.optim.lr_scheduler.LinearLR'>], 'scheduler_kwargs': {'start_factor': [1], 'end_factor': [0.01], 'total_iters': [50]}}]`


#### GIN model

- Config: `gin_config.py`
- Script: `run_experiment.py` with `gin` as `model_type`

Results are reported for this configuration:

`[{'node_features': [<NodeFeatures.ONE_HOT_REGION: 2>, <NodeFeatures.FC_MATRIX_ROW: 1>], 'correlation_type': [<CorrelationType.PEARSON: 'pearson'>], 'batch_size': [32], 'num_hidden_features': [45], 'num_sublayers': [2], 'dropout': [0.5], 'eps': [0.0], 'graph_kwargs': {'thresholding_function': [<ThresholdingFunction.SUBJECT_VALUES: 2>], 'threshold_by_absolute_value': [False], 'return_absolute_value': [False], 'threshold_type': [<DataThresholdingType.KNN: 1>], 'threshold': [5, 8, 10]}, 'criterion': [<function mse_loss at 0x0000019E119B3F78>], 'optimizer': [<class 'torch.optim.adamw.AdamW'>], 'optimizer_kwargs': {'lr': [0.003], 'weight_decay': [0.0001]}, 'epochs': [100], 'scheduler': [<class 'torch.optim.lr_scheduler.LinearLR'>], 'scheduler_kwargs': {'start_factor': [1], 'end_factor': [0.01], 'total_iters': [50]}}]`

## Sparsification experiment

- Config: `gin_config.py`
- Script: `run_experiment.py` with `gin` as `model_type`, and `--single_select_fold` option along with `num_select_folds 10` and `num_assess_folds 100`

The configuration is too long for printing. As such it is available in `sparsification_config.py` (still it needs to be run as `gin_config.py`).
