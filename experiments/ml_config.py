from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

cls_estimator_map = {
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "elastic_net": LogisticRegression,
}

cls_hyperparameters_map = {
    "knn": {
        "n_neighbors": [1, 2, 3, 4, 5],
        "p": [1, 2, 3],
        "weights": ["uniform", "distance"],
    },
    "naive_bayes": {
        "var_smoothing": [1e-15, 1e-10, 1e-9, 1e-8, 1e-5, 1e-2, 1e-1]
    },
    "svm": {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [1, 2, 3, 5, None],
        "criterion": ["entropy", "gini"],
    },
    "elastic_net": {
        "solver": ["saga"],
        "C": [0.1, 1, 10, 100, 1000],
        "penalty": ["elasticnet"],
        "l1_ratio": [0.15, 0.25, 0.5, 0.7],
        "max_iter": [10000],
    },
}

reg_estimator_map = {
    "knn": KNeighborsRegressor,
    "svm": SVR,
    "random_forest": RandomForestRegressor,
    "elastic_net": ElasticNet,
}

reg_hyperparameters_map = {
    "knn": {
        "n_neighbors": [1, 2, 3, 4, 5],
        "p": [1, 2, 3],
        "weights": ["uniform", "distance"],
    },
    "svm": {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["poly", "rbf", "sigmoid"],
        "degree": [1, 2, 3, 4],
        "epsilon": [1e-3, 1e-2, 1e-1],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [1, 2, 3, 5, None],
        "criterion": ["mse"],
    },
    "elastic_net": {
        "alpha": [0.01, 0.1, 1],
        "l1_ratio": [0.15, 0.25, 0.5, 0.7],
        "max_iter": [100000],
    },
}
