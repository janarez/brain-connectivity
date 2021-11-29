"""
Collection of all enums.

We extract all defined enums here, to enable easy debugging in Jupyter
using the `reload` module function.
"""

from enum import Enum, auto


class ConnectivityMode(Enum):
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


class CorrelationType(Enum):
    "Different types of correlation for the raw timeseries."
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    PARTIAL_PEARSON = "partial_pearson"

    def __str__(self):
        return f"{self.value}"


class DataThresholdingType(Enum):
    """
    Determines method of creating graph from a FC matrix.

    KNN: Take top `k` edges of each node.
    FIXED_THRESHOLD: Take all edges crossing a threshold.
    SPARSITY: Take top `x` % of edges from full matrix.
    """

    KNN = auto()
    FIXED_THRESHOLD = auto()
    # TODO: Sparsity is not implemented.
    SPARSITY = auto()


class ThresholdingFunction(Enum):
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


class NodeFeatures(Enum):
    """
    FC_MATRIX_ROW: Each node has as its features its row from FC matrix.
    ONE_HOT_REGION: Each node has as its features one hot encoding of its id.
    """

    FC_MATRIX_ROW = auto()
    ONE_HOT_REGION = auto()
