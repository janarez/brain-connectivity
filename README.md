# Brain connectivity repository

This repository contains code to run machine learning experiments with functional connectivity matrices, including training scripts with nested cross validation.

## Contents

- `brain-connectivity/`: Package with code powering the experiments.
  - `README.md`: Info about the package.
- `data-exploration/`
  - `timeseries.ipynb`: Preprocessing raw data for experiments. Exploration of raw timeseries.
  - `functional-connectivity.ipynb`: Visual exploration of functional connectivity matrices.
- `data/`: All data goes here. FC matrices are cached after first computation.
  - `timeseries.pickle`: Pickled numpy array of shape `[num_subjects, num_brain_regions, num_time_points]`.
  - `subjects.csv`: CSV with modeling targets that presumes header and index. Targets in header are {"target", "age", "sex"}.
- `experiments/`
  - `run_experiment.py`: Script for deep learning experiments.
  - `run_ml_experiment.py`: Script for standard machine learning experiments.
  - *`<model>`*`_config.py`: Hyperparameter configuration for *\<model\>*.


## Data

fMRI data are private and therefore not provided. Instead, the `data` folder contains dummy time series and targets.

## Get started

Clone the repository.

Install the `brain-connectivity` package via pip from its top level directory (where `setup.py` is located):

```bash
pip install -e .
```

### GPU

If you want to run the experiments on a GPU, you have to manually install PyTorch and PyTorch Geometric with the correct CUDA support for you machine. See especially the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip-wheels).

### Surrogates

The `brain-connectivity` package supports creating surrogate time series using the [`nolitsa` repository](https://github.com/manu-mannattil/nolitsa). It, however, needs to be patched. If you want to use this functionality run the following steps:

```bash
git clone https://github.com/manu-mannattil/nolitsa.git
cd nolitsa
git checkout 40bef
git apply ../nolitsa.patch
pip install -e .
```

### Developing

The package uses isort, black and flake8 for consistent formatting (see `.vscode/settings.json` for used settings). For development purposed you can install these dependencies with:

```bash
pip install -e .[dev]
```

## Run experiments

For standard machine learning experiments run the `run_ml_experiment.py` script. For PyTorch deep learning models run the `run_experiment.py` script.

To see the required and optional script arguments use the `--help` flag.

Each model's hyperparameter configuration is in its config file (e.g., `gin_config.py`).
