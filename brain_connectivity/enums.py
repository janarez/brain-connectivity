from enum import Enum, auto

"""
We extract all defined enums here, to enable easy debugging in Jupyter using the `reload` module function.
"""


class ModelType(Enum):
    GRAPH = auto()
    DENSE = auto()


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
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    PARTIAL_PEARSON = 'partial_pearson'

    def __str__(self):
        return f'{self.value}'


class DataThresholdingType(Enum):
    KNN = auto()
    FIXED_THRESHOLD = auto()
    SPARSITY = auto()


class ThresholdingFunction(Enum):
    GROUP_AVERAGE = auto()
    SUBJECT_VALUES = auto()
    EXPLICIT_MATRIX = auto()
    RANDOM = auto()


class NodeFeatures(Enum):
    FC_MATRIX_ROW = auto()
    ONE_HOT_REGION = auto()
