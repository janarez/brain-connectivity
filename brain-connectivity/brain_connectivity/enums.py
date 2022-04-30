"""
Collection of all enums.

We extract all defined enums here, to enable easy debugging in Jupyter
using the `reload` module function.
"""

from enum import Enum, auto

import numpy as np

from .data_utils import granger_causality, xicorr


class CustomEnum(Enum):
    def __str__(self):
        return f"{self.name.lower()}"


class ConnectivityMode(CustomEnum):
    """
    Determines how is connectivity matrix obtained.

    FIXED: Use handmade FC matrix.
    START: Learn FC matrix only on raw input features.
    SINGLE: Learn FC matrix on raw input features as well as all subsequent feature mapping layers.
    MULTIPLE: Learn new FC matrix before every feature mapping layer.
    """

    FIXED = auto()
    START = auto()
    SINGLE = auto()
    MULTIPLE = auto()


class CorrelationType(CustomEnum):
    "Different types of correlation / causality for the raw timeseries."
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    # Need to wrap function to make it an attribute, not method definition.
    XI = np.vectorize(xicorr, signature="(n),(n)->()")
    GRANGER = np.vectorize(granger_causality, signature="(n),(n)->()")

    @property
    def is_symmetric(self):
        return (
            False
            if self is CorrelationType.GRANGER or self is CorrelationType.XI
            else True
        )


class DataThresholdingType(CustomEnum):
    """
    Determines method of creating graph from a FC matrix.

    KNN: Take top `k` edges of each node.
    FIXED_THRESHOLD: Take all edges crossing a threshold.
    SPARSITY: Take top `x` % of edges from full matrix.
    """

    KNN = auto()
    FIXED_THRESHOLD = auto()
    SPARSITY = auto()


class ThresholdingFunction(CustomEnum):
    """
    Determines based on what information are graphs created.

    GROUP_AVERAGE: Threshold using matrix created as average difference between the two groups.
    SUBJECT_VALUES: Threshold on matrix of each subject.
    EXPLICIT_MATRIX: Threshold on some given matrix instead of data.
    RANDOM: Threshold on random FC matrix instead of data.
    """

    GROUP_AVERAGE = auto()
    SUBJECT_VALUES = auto()
    EXPLICIT_MATRIX = auto()
    RANDOM = auto()


class NodeFeatures(CustomEnum):
    """
    FC_MATRIX_ROW: Each node has as its features its row from FC matrix.
    ONE_HOT_REGION: Each node has as its features one hot encoding of its id.
    """

    FC_MATRIX_ROW = auto()
    ONE_HOT_REGION = auto()
    ONE = auto()
    ONE_HOT_CAT_FC_ROW = auto()
