import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from brain_connectivity import data_utils, dataset, enums, general_utils
from sklearn.model_selection import GridSearchCV

from ml_config import (
    cls_estimator_map,
    cls_hyperparameters_map,
    reg_estimator_map,
    reg_hyperparameters_map,
)


def main(args):
    os.makedirs(args.experiment_folder, exist_ok=False)
    exp_logger = general_utils.get_logger(
        "experiment",
        os.path.join(args.experiment_folder, "experiment.txt"),
    )

    # Get targets.
    df = pd.read_csv(
        os.path.join(args.data_folder, "subjects.csv"),
        index_col=0,
    )
    targets = df[args.target_column].values

    # Init cross-validation.
    cv = data_utils.NestedCrossValidation(
        targets=targets,
        num_assess_folds=args.num_assess_folds,
        num_select_folds=args.num_select_folds,
        random_state=args.random_cv_seed,
        stratified=not args.regression_experiment,
    )

    # Set task type related variables.
    estimator_map = (
        reg_estimator_map if args.regression_experiment else cls_estimator_map
    )
    estimator_cls = estimator_map[args.estimator]
    hyperparameters_map = (
        reg_hyperparameters_map
        if args.regression_experiment
        else cls_hyperparameters_map
    )
    hyperparameters = hyperparameters_map[args.estimator]
    eval_metric = "rmse" if args.regression_experiment else "accuracy"

    # Experiment results.
    exp_test_results = defaultdict(list)

    # Dataset.
    data = dataset.FunctionalConnectivityDataset(
        args.experiment_folder,
        args.data_folder,
        None,
        targets=targets,
        correlation_type=enums.CorrelationType.PEARSON,
    )

    for outer_id in cv.outer_cross_validation():
        os.makedirs(
            os.path.join(args.experiment_folder, f"{outer_id:03d}"),
            exist_ok=False,
        )
        logger = general_utils.get_logger(
            "cv",
            os.path.join(args.experiment_folder, f"{outer_id:03d}", "cv.txt"),
        )
        logger.info(f"Outer fold {outer_id+1} / {args.num_assess_folds}")

        scoring = (
            {
                "rmse": metrics.make_scorer(
                    lambda x, y: metrics.mean_squared_error(x, y) ** 0.5,
                    greater_is_better=False,
                ),
                "mae": metrics.make_scorer(
                    metrics.mean_absolute_error, greater_is_better=False
                ),
            }
            if args.regression_experiment
            else {
                "accuracy": metrics.make_scorer(metrics.accuracy_score),
                "precision": metrics.make_scorer(
                    metrics.precision_score, zero_division=0
                ),
                "recall": metrics.make_scorer(metrics.recall_score),
            }
        )

        grid = GridSearchCV(
            estimator_cls(),
            hyperparameters,
            cv=args.num_select_folds,
            n_jobs=-1,
            scoring=scoring,
            # Fits `grid.best_estimator_` on full `dev` dataset.
            refit=eval_metric,
        )
        grid.fit(
            *data.ml_loader(
                dataset="dev", indices=cv.dev_indices, flatten=args.flatten
            )
        )
        # Validation results.
        cv_params = ["params"] + [
            f"{s}_test_{k}" for k in scoring.keys() for s in ["mean", "std"]
        ]
        val_results = pd.DataFrame(grid.cv_results_)
        val_results["rank"] = val_results[f"mean_test_{eval_metric}"]
        val_results = val_results.sort_values(by=["rank"], ascending=False)[
            cv_params
        ]
        for k in scoring.keys():
            logger.info(
                f"Val {k}: {np.abs(val_results[f'mean_test_{k}'].iloc[0]):.4f} ± {val_results[f'std_test_{k}'].iloc[0]:.4f}"
            )

        logger.info(f"Best hyperparameters: {grid.best_params_}")
        # Test results.
        test_results = defaultdict(list)
        for _ in range(3):
            X, y = data.ml_loader(
                dataset="test", indices=cv.test_indices, flatten=args.flatten
            )
            y_pred = grid.best_estimator_.predict(X)
            for k, scorer in scoring.items():
                test_results[k].append(scorer._score_func(y, y_pred))

        for k in scoring.keys():
            exp_test_results[k].append(np.mean(test_results[k]))
            logger.info(
                f"Test {k}: {np.mean(test_results[k]):.4f} ± {np.std(test_results[k]):.4f}"
            )
        general_utils.close_logger("cv")

    for k in scoring.keys():
        exp_logger.info(
            f"Exp {k}: {np.mean(exp_test_results[k]):.4f} ± {np.std(exp_test_results[k]):.4f}"
        )
    general_utils.close_all_loggers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs two stage cross validation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment_folder",
        help="Folder for saving experiment logs.",
    )
    parser.add_argument(
        "estimator",
        help="Model to run. Naive Bayes cannot be run with --regression_experiment flag and elastic net without it.",
        choices=["knn", "naive_bayes", "svm", "random_forest", "elastic_net"],
    )
    parser.add_argument(
        "target_column",
        help="The predicted variable. Use --regression_experiment flag for 'age'.",
        choices=["target", "sex", "age"],
    )
    parser.add_argument(
        "flatten",
        help="How to flatten data matrices.",
        choices=["flattened-dense", "triangular-dense"],
        default="flattened-dense",
        nargs="?",
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
        "--regression_experiment",
        help="Run regression instead of binary classification task.",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
