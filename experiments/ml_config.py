from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
