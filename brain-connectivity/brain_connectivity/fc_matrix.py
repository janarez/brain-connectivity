"""
Implementations for several FC matrix sparsification strategies.
The main `create_connectivity_matrices` function dispatches to all the specific methods.
"""


import operator
from typing import Optional, Union

import numpy as np

from .enums import DataThresholdingType, ThresholdingFunction


def create_connectivity_matrices(
    data: np.array,
    thresholding_function: ThresholdingFunction,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Union[operator.le, operator.ge] = operator.ge,
    threshold_by_absolute_value: bool = False,
    return_absolute_value: bool = False,
    # Specific to `ThresholdingFunction.GROUP_AVERAGE` and `ThresholdingFunction.EXPLICIT_MATRIX`.
    thresholding_matrix=None,
    # Specific to `ThresholdingFunction.RANDOM`.
    per_subject: bool = True,
):
    """
    Sparsifies FC matrices (`data` [num_subjects, size, size]) by several available methods.

    Args:
        data (`np.array`): FC matrices of shape [num_subjects, size, size].
        thresholding_function (`ThresholdingFunction`): Based on what data are FC matrices thresholded.
        threshold_type (`DataThresholdingType`): Sparsification strategy.
        threshold: (`Union[float, int]`): How much / many edges to preserve.
        thresholding_operator (`Union[operator.le, operator.ge]`): Whether to take greatest (ge)
            or lowest (le) values. Default `operator.ge`.
        threshold_by_absolute_value (`bool`): If the thresholding is done assuming absolute values. Default `False`.
        return_absolute_value (`bool`): If the returned sparsified matrices are in absolute value. Default `False`.
        thresholding_matrix (`np.array`): For `ThresholdingFunction.GROUP_AVERAGE`
            and `ThresholdingFunction.EXPLICIT_MATRIX`. Matrix [size, size] to compute the sparsification from.
        per_subject (`bool`): For `ThresholdingFunction.RANDOM`, if the random matrix
            is different for each sample. Default `True`.

    Returns:
        (`np.array`, `np.array`): Sparsified `data`, first array is binary, second real valued.
    """
    if thresholding_function == ThresholdingFunction.GROUP_AVERAGE:
        b, r = _get_data_threshold_at_largest_average_difference_between_groups(
            raw_fc_matrices=data,
            avg_difference_matrix=thresholding_matrix,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
        )
    elif thresholding_function == ThresholdingFunction.SUBJECT_VALUES:
        b, r = _get_data_thresholded_by_sample_values(
            raw_fc_matrices=data,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
            threshold_by_absolute_value=threshold_by_absolute_value,
            return_absolute_value=return_absolute_value,
        )
    elif thresholding_function == ThresholdingFunction.EXPLICIT_MATRIX:
        b, r = _get_data_thresholded_by_explicit_matrix(
            raw_fc_matrices=data,
            thresholding_matrix=thresholding_matrix,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
            threshold_by_absolute_value=threshold_by_absolute_value,
            return_absolute_value=return_absolute_value,
        )
    elif thresholding_function == ThresholdingFunction.RANDOM:
        b, r = _get_data_thresholded_by_random_matrix(
            raw_fc_matrices=data,
            per_subject=per_subject,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
            threshold_by_absolute_value=threshold_by_absolute_value,
            return_absolute_value=return_absolute_value,
        )
    # Binarized and real valued matrices.
    return b, r


def get_matrix_of_avg_diff_between_groups(
    raw_fc_matrices: np.array,
    binary_targets: np.array,
    train_indices: np.array,
):
    """
    Computes difference between average FC matrix for group 0 and group 1
    taking into account only indexes given by `train_indices`.

    Args:
        raw_fc_matrices (`np.array`): FC matrices of shape [num_subjects, size, size].
        binary_targets (`np.array`): 0/1 targets of shape [num_subjects].
        train_indices (`np.array`): Indices for subset subjects from `raw_fc_matrices`.

    Returns:
        (`np.array`): The difference matric of shape [size, size].
    """
    # Base average information only on training data.
    zero_train_indices = train_indices[binary_targets[train_indices] == 0]
    one_train_indices = train_indices[binary_targets[train_indices] == 1]

    # Take average matrix for both groups.
    avg_zero_matrix = np.mean(raw_fc_matrices[zero_train_indices], axis=0)
    avg_one_matrix = np.mean(raw_fc_matrices[one_train_indices], axis=0)
    # Average between groups difference.
    avg_difference_matrix = np.abs(avg_zero_matrix - avg_one_matrix)

    return avg_difference_matrix


def _get_data_threshold_at_largest_average_difference_between_groups(
    raw_fc_matrices,
    avg_difference_matrix,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
):
    avg_difference_matrix = np.repeat(
        np.expand_dims(avg_difference_matrix, 0),
        repeats=raw_fc_matrices.shape[0],
        axis=0,
    )

    return _get_data_thresholded_by_matrix(
        raw_fc_matrices,
        avg_difference_matrix,
        threshold_type,
        threshold,
        thresholding_operator,
        threshold_by_absolute_value=True,
        return_absolute_value=True,
    )


def _get_data_thresholded_by_sample_values(
    raw_fc_matrices,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
    threshold_by_absolute_value: bool,
    return_absolute_value: bool,
):
    return _get_data_thresholded_by_matrix(
        raw_fc_matrices,
        raw_fc_matrices,
        threshold_type,
        threshold,
        thresholding_operator,
        threshold_by_absolute_value,
        return_absolute_value,
    )


def _get_data_thresholded_by_explicit_matrix(
    raw_fc_matrices,
    thresholding_matrix,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
    threshold_by_absolute_value: bool,
    return_absolute_value: bool,
):
    # Expand over all samples.
    if len(thresholding_matrix.shape) == 2:
        thresholding_matrix = np.repeat(
            np.expand_dims(thresholding_matrix, 0),
            repeats=raw_fc_matrices.shape[0],
            axis=0,
        )

    return _get_data_thresholded_by_matrix(
        raw_fc_matrices,
        thresholding_matrix,
        threshold_type,
        threshold,
        thresholding_operator,
        threshold_by_absolute_value,
        return_absolute_value,
    )


def _get_data_thresholded_by_random_matrix(
    raw_fc_matrices,
    per_subject: bool,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
    threshold_by_absolute_value: bool,
    return_absolute_value: bool,
):
    num_subjects, num_regions, _ = raw_fc_matrices.shape

    # Generate random thresholding matrix with values from [-1, 1].
    if per_subject:
        thresholding_matrix = (
            2 * np.random.random((num_subjects, num_regions, num_regions)) - 1
        )
    else:
        thresholding_matrix = (
            2 * np.random.random((1, num_regions, num_regions)) - 1
        )
        thresholding_matrix = np.repeat(
            thresholding_matrix, repeats=num_subjects, axis=0
        )

    return _get_data_thresholded_by_matrix(
        raw_fc_matrices,
        thresholding_matrix,
        threshold_type,
        threshold,
        thresholding_operator,
        threshold_by_absolute_value,
        return_absolute_value,
    )


def _get_data_thresholded_by_matrix(
    raw_fc_matrices,
    thresholding_matrix,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
    threshold_by_absolute_value: bool,
    return_absolute_value: bool,
):
    assert raw_fc_matrices.shape == thresholding_matrix.shape
    num_subjects, num_regions, _ = raw_fc_matrices.shape

    fc = (
        np.abs(thresholding_matrix)
        if threshold_by_absolute_value
        else thresholding_matrix
    )
    raw_fc_matrices = (
        np.abs(raw_fc_matrices) if return_absolute_value else raw_fc_matrices
    )

    # Compute mask.
    if threshold_type == DataThresholdingType.FIXED_THRESHOLD:
        assert isinstance(
            threshold, float
        ), f"Used {type(threshold)} instead of `float` for {DataThresholdingType.FIXED_THRESHOLD}."
        mask = np.where(thresholding_operator(fc, threshold), True, False)
    elif threshold_type == DataThresholdingType.KNN:
        assert isinstance(
            threshold, int
        ), f"Used {type(threshold)} instead of `int` for {DataThresholdingType.KNN}."
        mask = np.zeros((num_subjects, num_regions, num_regions), dtype=bool)

        # Take top `threshold` neighbors.
        if thresholding_operator is operator.ge:
            knn_index = np.argsort(fc)[:, :, -threshold:]
        # Take lowest `threshold` neighbors.
        else:
            knn_index = np.argsort(fc)[:, :, :threshold]
        # Mark selected in mask.
        for s in range(num_subjects):
            for r in range(num_regions):
                mask[s, r, knn_index[s, r]] = True
    elif threshold_type == DataThresholdingType.SPARSITY:
        assert (
            isinstance(threshold, int) and threshold >= 0 and threshold < 100
        ), f"Threshold for {DataThresholdingType.SPARSITY} must be int in [0, 100)."
        th_per_matrix = np.percentile(
            fc, q=(100 - threshold), axis=(1, 2)
        ).reshape(-1, 1, 1)
        mask = np.where(thresholding_operator(fc, th_per_matrix), True, False)

    # Transform raw data.
    binary_fc_matrices = np.where(mask, 1, 0)
    real_fc_matrices = np.where(mask, raw_fc_matrices, 0)

    return binary_fc_matrices, real_fc_matrices
