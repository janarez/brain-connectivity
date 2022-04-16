import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from brain_connectivity import data_utils, dataset, general_utils, training
from tqdm import tqdm


def expand_config(model_type):
    # isort: off
    global model_class
    global config
    if model_type == "gin":
        from brain_connectivity.models.graph import GIN as model_class
        import gin_config as config
    elif model_type == "gat":
        from brain_connectivity.models.graph import GAT as model_class
        import gat_config as config
    elif model_type == "connectivity-dense":
        from brain_connectivity.models.dense import (
            ConnectivityDenseNet as model_class,
        )
        import connectivity_dense_config as config
    elif model_type in [
        "flattened-dense",
        "triangular-dense",
    ]:
        from brain_connectivity.models.dense import DenseNet as model_class
        import dense_config as config
    else:
        raise ValueError(
            f"`model_type` ({model_type}) not mapped to class and hyperparams"
        )
    # isort: on


def collect_results(results, next_result, key):
    for k in ["accuracy", "recall", "precision"]:
        results[k].append(key(next_result[k]))


def log_results(results, results_name, logger=logging):
    for k in ["accuracy", "recall", "precision"]:
        logger.info(
            f"{results_name.title()} {k}: {np.mean(results[k]):.4f} ± {np.std(results[k]):.4f}"
        )


def init_traning(
    log_folder,
    data_folder,
    device,
    hyperparameters,
    targets,
):
    # Prepare model.
    model_arguments = {
        **config.model_params,
        **{
            k: hyperparameters[k]
            for k in model_class.hyperparameters
            if k in hyperparameters.keys()
        },
    }
    model_class.log(log_folder, model_arguments)

    data = dataset.FunctionalConnectivityDataset(
        targets=targets,
        data_folder=data_folder,
        device=device,
        **{
            k: hyperparameters[k]
            for k in dataset.FunctionalConnectivityDataset.hyperparameters
            if k in hyperparameters.keys()
        },
        log_folder=log_folder,
    )

    trainer = training.Trainer(
        **config.training_params,
        **{
            k: hyperparameters[k]
            for k in training.Trainer.hyperparameters
            if k in hyperparameters.keys()
        },
        log_folder=log_folder,
    )

    return model_arguments, data, trainer


def model_type_to_dataloader_view(model_type):
    if model_type in ["gin", "gat"]:
        return "graph"
    elif model_type in [
        "connectivity-dense",
        "flattened-dense",
        "triangular-dense",
    ]:
        return model_type
    else:
        raise ValueError(
            f"`model_type` ({model_type}) not mapped to dataloader view."
        )


def main(args):
    expand_config(args.model_type)

    os.makedirs(args.experiment_folder, exist_ok=False)
    exp_logger = general_utils.get_logger(
        "experiment",
        os.path.join(args.experiment_folder, "experiment.txt"),
    )

    # Select device.
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    exp_logger.info(f"Device: {device}")
    exp_logger.info(f"Random CV seed: {args.random_cv_seed}")
    exp_logger.info(f"Random model seed: {args.random_model_seed}")

    # Get targets.
    df = pd.read_csv(
        os.path.join(args.data_folder, "patients-cleaned.csv"),
        index_col=0,
    )
    targets = df[args.target_column].values

    # Init cross-validation.
    cv = data_utils.StratifiedCrossValidation(
        targets=targets,
        num_assess_folds=args.num_assess_folds,
        num_select_folds=args.num_select_folds,
        random_state=args.random_cv_seed,
        single_select_fold=args.single_select_fold,
    )

    # Experiment results.
    exp_dev_results = defaultdict(list)
    exp_test_results = defaultdict(list)

    for outer_id in cv.outer_cross_validation():
        general_utils.set_model_random_state(args.random_model_seed)
        os.makedirs(
            os.path.join(args.experiment_folder, f"{outer_id:03d}"),
            exist_ok=False,
        )
        logger = general_utils.get_logger(
            "cv",
            os.path.join(args.experiment_folder, f"{outer_id:03d}", "cv.txt"),
        )
        logger.info(f"Outer fold {outer_id+1} / {args.num_assess_folds}")

        # Model selection.
        # Keep best hyperparameters.
        best_hyperparameters = None
        best_mean_accuracy = 0
        best_std_accuracy = 0

        hyperparameter_grid = data_utils.DoubleLevelParameterGrid(
            config.hyperparameters
        )
        exp_logger.info(
            f"Selecting from {len(hyperparameter_grid)} hyperparameters."
        )
        exp_logger.debug(hyperparameter_grid.orig_param_grid)

        for hyper_id, hyperparameters in enumerate(hyperparameter_grid):
            logger.info(f"Evaluating hyperparameters #{hyper_id:04d}")
            log_folder = os.path.join(
                args.experiment_folder,
                f"{outer_id:03d}",
                f"{hyper_id:04d}_{training.stringify(hyperparameters)}",
            )
            os.makedirs(log_folder, exist_ok=False)

            model_arguments, data, trainer = init_traning(
                log_folder,
                args.data_folder,
                device,
                hyperparameters,
                targets,
            )

            # Run training.
            train_dataset = "train"
            eval_dataset = "val"
            for inner_id in tqdm(
                cv.inner_cross_validation(),
                total=args.num_select_folds,
                desc="Inner CV",
                leave=False,
            ):
                logger.debug(
                    f"Inner fold {inner_id+1} / {args.num_select_folds}"
                )
                trainer.train(
                    model=model_class(**model_arguments).to(device),
                    named_trainloader=(
                        train_dataset,
                        # TODO: Variable loaders.
                        data.dataloader(
                            dataset=train_dataset,
                            indices=cv.train_indices,
                            view=model_type_to_dataloader_view(args.model_type),
                        ),
                    ),
                    named_evalloader=(
                        eval_dataset,
                        data.dataloader(
                            dataset=eval_dataset,
                            indices=cv.val_indices,
                            view=model_type_to_dataloader_view(args.model_type),
                        ),
                    ),
                    fold=inner_id,
                )

            # Results.
            train_results, eval_results = trainer.get_results(
                train_dataset=train_dataset, eval_dataset=eval_dataset
            )
            logger.debug(f"Train: {train_results}")
            logger.debug(f"Val: {eval_results}")

            # Update best setting based on eval accuracy
            max_mean_accuracy = eval_results["accuracy"][0][-1]
            max_std_accuracy = eval_results["accuracy"][1][-1]
            logger.info(
                f"Val accuracy: {max_mean_accuracy:.4f} ± {max_std_accuracy:.4f}"
            )

            if max_mean_accuracy > best_mean_accuracy:
                best_hyperparameters = hyperparameters
                best_mean_accuracy = max_mean_accuracy
                best_std_accuracy = max_std_accuracy
                logger.info(
                    f"New best val accuracy: {best_mean_accuracy:.4f} ± {best_std_accuracy:.4f}"
                )
                logger.info(f"New best hyperparameters: {best_hyperparameters}")

        # Model assessment.
        exp_logger.info(f"Best hyperparameters: {best_hyperparameters}")
        exp_logger.info(
            f"Best accuracy: {best_mean_accuracy:.4f} ± {best_std_accuracy:.4f}"
        )
        if args.single_select_fold:
            return

        # Average over 3 runs to offset random initialization.
        test_results = defaultdict(list)
        dev_results = defaultdict(list)
        for test_id in range(3):
            general_utils.set_model_random_state(None)
            log_folder = os.path.join(
                args.experiment_folder, f"{outer_id:03d}", f"test_{test_id}"
            )
            os.makedirs(log_folder, exist_ok=False)

            model_arguments, data, trainer = init_traning(
                log_folder,
                args.data_folder,
                device,
                best_hyperparameters,
                targets,
            )
            # Run training.
            train_dataset = "dev"
            eval_dataset = "test"
            trainer.train(
                model=model_class(**model_arguments).to(device),
                named_trainloader=(
                    train_dataset,
                    data.dataloader(
                        dataset=train_dataset,
                        indices=cv.dev_indices,
                        view=model_type_to_dataloader_view(args.model_type),
                    ),
                ),
                named_evalloader=(
                    eval_dataset,
                    data.dataloader(
                        dataset=eval_dataset,
                        indices=cv.test_indices,
                        view=model_type_to_dataloader_view(args.model_type),
                    ),
                ),
                fold=0,
            )
            # Results.
            train_results, eval_results = trainer.get_results(
                train_dataset=train_dataset, eval_dataset=eval_dataset
            )
            collect_results(dev_results, train_results, key=lambda x: x[0][-1])
            collect_results(test_results, eval_results, key=lambda x: x[0][-1])

        log_results(dev_results, "dev", exp_logger)
        log_results(test_results, "test", exp_logger)

        collect_results(exp_dev_results, dev_results, key=lambda x: np.mean(x))
        collect_results(
            exp_test_results, test_results, key=lambda x: np.mean(x)
        )
        general_utils.close_logger("cv")

    log_results(exp_dev_results, "exp dev", exp_logger)
    log_results(exp_test_results, "exp test", exp_logger)
    general_utils.close_all_loggers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs two stage cross validation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment_folder",
        help="Folder for saving experiment data (logs, TensorBoard).",
    )
    parser.add_argument(
        "model_type",
        help="Model to run.",
        choices=[
            "gin",
            "gat",
            "connectivity-dense",
            "flattened-dense",
            "triangular-dense",
        ],
    )
    parser.add_argument(
        "target_column",
        help="The predicted variable.",
        choices=["target", "sex"],
    )
    parser.add_argument(
        "--data_folder",
        help="Folder with raw dataset.",
        default=os.path.normpath("../data"),
        nargs="?",
    )
    parser.add_argument(
        "--num_assess_folds",
        help="Number of folds for outter cross validation loop.",
        type=int,
        default=10,
        nargs="?",
    )
    parser.add_argument(
        "--num_select_folds",
        help="Number of folds for inner cross validation loop.",
        type=int,
        default=3,
        nargs="?",
    )
    parser.add_argument(
        "--random_cv_seed",
        help="Random seed for cross validation.",
        type=int,
        default=0,
        nargs="?",
    )
    parser.add_argument(
        "--random_model_seed",
        help="Random seed for model initialization.",
        type=int,
        default=0,
        nargs="?",
    )
    parser.add_argument(
        "--use_gpu",
        help="Whether GPU should be used. Fallbacks to 'cpu'.",
        action="store_true",
    )
    parser.add_argument(
        "--single_select_fold",
        help="For experimentation purposes. Returns after single select fold.",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
