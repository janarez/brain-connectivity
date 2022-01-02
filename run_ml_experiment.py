import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from brain_connectivity import data_utils, dataset, enums, general_utils

estimator_map = {
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "elastic_net": SGDClassifier,
}

hyperparameters_map = {
    "knn": {
        "n_neighbors": [1, 2, 3, 4, 5, 7, 10],
        "p": [1, 2],
        "weights": ("uniform", "distance"),
    },
    "naive_bayes": {
        "var_smoothing": [1e-15, 1e-10, 1e-9, 1e-8, 1e-5, 1e-2, 1e-1]
    },
    "svm": {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ("poly", "rbf", "sigmoid"),
        "degree": [1, 2, 3, 4],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [1, 2, 3, 5, None],
        "criterion": ("entropy", "gini"),
    },
    "elastic_net": {
        "loss": ["log", "modified_huber"],
        "alpha": [0.1, 0.01, 0.001],
        "penalty": ["elasticnet"],
        "l1_ratio": [0.15, 0.25, 0.5, 0.7],
        "max_iter": [1000, 10000, 20000],
    },
}

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
        help="Model to run.",
        choices=["knn", "naive_bayes", "svm", "random_forest", "elastic_net"],
    )
    parser.add_argument(
        "target_column",
        help="The predicted variable.",
        choices=["target", "sex"],
    )
    parser.add_argument(
        "flatten",
        help="How to flatten data matrices.",
        choices=["all", "triag"],
        default="all",
        nargs="?",
    )
    parser.add_argument(
        "--data_folder",
        help="Folder with raw dataset.",
        default=os.path.normpath("./data"),
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
        default=10,
        nargs="?",
    )
    parser.add_argument(
        "--random_cv_seed",
        help="Random seed for cross validation.",
        type=int,
        default=None,
        nargs="?",
    )
    args = parser.parse_args()

    os.makedirs(args.experiment_folder, exist_ok=False)
    exp_logger = general_utils.get_logger(
        "experiment",
        os.path.join(args.experiment_folder, "experiment.txt"),
    )

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
    )

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

        scoring = {
            "accuracy": metrics.make_scorer(metrics.accuracy_score),
            "precision": metrics.make_scorer(
                metrics.precision_score, zero_division=0
            ),
            "recall": metrics.make_scorer(metrics.recall_score),
        }

        grid = GridSearchCV(
            estimator_map[args.estimator](),
            hyperparameters_map[args.estimator],
            cv=args.num_select_folds,
            n_jobs=-1,
            scoring=scoring,
            refit="accuracy",
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
        val_results["rank"] = (
            val_results["mean_test_accuracy"] - val_results["std_test_accuracy"]
        )
        val_results = val_results.sort_values(by=["rank"], ascending=False)[
            cv_params
        ]
        for k in scoring.keys():
            logger.info(
                f"Val {k}: {val_results[f'mean_test_{k}'][0]:.4f} ± {val_results[f'std_test_{k}'][0]:.4f}"
            )

        # Test results.
        test_results = defaultdict(list)
        for _ in range(3):
            X, y = data.ml_loader(
                dataset="test", indices=cv.test_indices, flatten="all"
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
