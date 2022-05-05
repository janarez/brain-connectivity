# Brain connectivity package

Source code to run machine learning experiments with functional connectivity matrices.

## Installation

Install the package via pip from its top level directory (where `setup.py` is located):

```bash
pip install -e .
```

## Contents

- `models/`
  - `model.py`: Base model class. All models need to inherit from it.
  - `dense.py`: Graph neural networks.
  - `graph.py`: Feed forward models.
- `data_utils.py`: Collection of data related helpers. Includes extension of sklearn's `ParameterGrid` and nested cross validation.
- `dataset.py`: Dataset class that takes care of computing FC matrices, correct data format for different models and creating dataloaders.
- `enums.py`: Collection of enums mostly for defining FC matrix processing choices.
- `evaluation.py`: Classes for calculating metrics and logging them to Tensorboard.
- `fc_matrix.py`: Implementations for several FC matrix sparsification strategies.
- `general_utils.py`: Logging and random state utils.
- `training.py`: General training class for training PyTorch models.
- `visualizations.py`: Matplotlib and seaborn visualizations.
