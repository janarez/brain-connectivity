# Brain connectivity repository

This repository contains code to run machine learning experiments with functional connectivity matrices, including training scripts with nested cross validation.

You can read my 2022 Master thesis from MMF UK in Prague on "Utilization of brain connectivity in classification and regression tasks in brain data" [here](https://dspace.cuni.cz/handle/20.500.11956/173962).

## Contents

- `brain-connectivity/`: Package with code powering the experiments.
  - [`README.md`](./brain-connectivity/README.md): Info about the package.
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

```bash
git clone https://github.com/janarez/brain-connectivity.git
```

Since installing PyTorch and PyTorch Geometric using the correct wheels from `setup.py` is tricky, the whole repository has its own `requirements.txt` file. Install requirements by running:

```bash
cd brain-connectivity
pip install -r requirements.txt
```

You might need to adjust the CUDA version for your machine (or don't use it).

Then install the `brain-connectivity` package via pip from its top level directory (where `setup.py` is located):

```bash
cd brain-connectivity
pip install -e .
```

### Surrogates

The `brain-connectivity` package supports creating surrogate time series using the [`nolitsa` repository](https://github.com/manu-mannattil/nolitsa). It, however, needs to be patched. If you want to use this functionality run the following steps:

```bash
cd ..
git clone https://github.com/manu-mannattil/nolitsa.git
cd nolitsa
git checkout 40bef
git apply ../nolitsa.patch
pip install -e .
```

### Developing

The `brain_connectivity` package uses isort, black and flake8 for consistent formatting (see `.vscode/settings.json` for used settings). For development purposes you might want to include these dependencies during package install by:

```bash
pip install -e .[dev]
```

## Run experiments

For standard machine learning experiments run the `run_ml_experiment.py` script. For PyTorch deep learning models run the `run_experiment.py` script.

To see the required and optional script arguments use the `--help` flag.

Each model's hyperparameter configuration is in its config file (e.g., `gin_config.py`). The hyperparameter names correspond to initialization arguments of either model, trainer or dataset class. You can check the relevant class for description of each hyperparameter.

If you want to replicate the experiments from the diploma thesis read [`thesis_experiments.md`](./experiments/thesis_experiments.md). The [frozen requirements](./experiments/thesis_frozen_requirements.txt) might come in handy too.

## Notebooks

The `data-exploration` folder contains two notebooks. They are saved with outputs, so you can check what functional connectivity data look like.

The `timeseries.ipynb` notebook in the first part loads raw data and prepares them for experiments. This can only be run with the raw data files. The second part visualizes the time series and can be run independently of the first part.

The `functional-connectivity.ipynb` notebook explores functional connectivity matrices. You can create the matrices using connectivity functions from the `brain_connectivity` package.
