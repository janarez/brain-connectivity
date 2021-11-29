import operator
from typing import Optional, Union

import numpy as np

from .enums import DataThresholdingType, ThresholdingFunction

rng = np.random.default_rng(42)


def create_connectivity_matrices(
    data,
    thresholding_function: ThresholdingFunction,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[
        Union[operator.le, operator.ge]
    ] = operator.ge,
    threshold_by_absolute_value: bool = True,
    return_absolute_value: bool = False,
    # Specific to `ThresholdingFunction.GROUP_AVERAGE` and `ThresholdingFunction.EXPLICIT_MATRIX`.
    thresholding_matrix=None,
    # Specific to `ThresholdingFunction.RANDOM`.
    per_subject: bool = True,
):
    if thresholding_function == ThresholdingFunction.GROUP_AVERAGE:
        b, r = get_data_threshold_at_largest_average_difference_between_groups(
            raw_fc_matrices=data,
            avg_difference_matrix=thresholding_matrix,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
        )
    elif thresholding_function == ThresholdingFunction.SUBJECT_VALUES:
        b, r = get_data_thresholded_by_sample_values(
            raw_fc_matrices=data,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
            threshold_by_absolute_value=threshold_by_absolute_value,
            return_absolute_value=return_absolute_value,
        )
    elif thresholding_function == ThresholdingFunction.EXPLICIT_MATRIX:
        b, r = get_data_thresholded_by_explicit_matrix(
            raw_fc_matrices=data,
            thresholding_matrix=thresholding_matrix,
            threshold_type=threshold_type,
            threshold=threshold,
            thresholding_operator=thresholding_operator,
            threshold_by_absolute_value=threshold_by_absolute_value,
            return_absolute_value=return_absolute_value,
        )
    elif thresholding_function == ThresholdingFunction.RANDOM:
        b, r = get_data_thresholded_by_random_matrix(
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
    raw_fc_matrices,
    binary_targets,
    train_indices,
):
    # Base average information only on training data.
    zero_train_indices = train_indices[binary_targets[train_indices] == 0]
    one_train_indices = train_indices[binary_targets[train_indices] == 1]

    # Take average matrix for both groups.
    avg_zero_matrix = np.mean(raw_fc_matrices[zero_train_indices], axis=0)
    avg_one_matrix = np.mean(raw_fc_matrices[one_train_indices], axis=0)
    # Average between groups difference.
    avg_difference_matrix = np.abs(avg_zero_matrix - avg_one_matrix)

    return avg_difference_matrix


def get_data_threshold_at_largest_average_difference_between_groups(
    raw_fc_matrices,
    avg_difference_matrix,
    threshold_type: DataThresholdingType,
    threshold: Union[float, int],
    thresholding_operator: Optional[Union[operator.le, operator.ge]],
):
    num_subjects, num_regions, _ = raw_fc_matrices.shape

    # Compute mask.
    if threshold_type == DataThresholdingType.FIXED_THRESHOLD:
        assert (
            type(threshold) == float
        ), f"Used {type(threshold)} instead of `float` for {DataThresholdingType.FIXED_THRESHOLD}."
        mask = np.where(
            thresholding_operator(avg_difference_matrix, threshold), True, False
        )
    elif threshold_type == DataThresholdingType.KNN:
        assert (
            type(threshold) == int
        ), f"Used {type(threshold)} instead of `int` for {DataThresholdingType.KNN}."
        mask = np.zeros((num_regions, num_regions), dtype=bool)
        # Take top `threshold` neighbors.
        if thresholding_operator is operator.ge:
            knn_index = np.argsort(avg_difference_matrix)[:, -threshold:]
        # Take lowest `threshold` neighbors.
        else:
            knn_index = np.argsort(avg_difference_matrix)[:, :threshold]
        # Mark selected in mask.
        for r in range(num_regions):
            mask[r, knn_index[r]] = True
    elif threshold_type == DataThresholdingType.SPARSITY:
        raise NotImplementedError()

    # Repeat mask.
    mask = np.tile(mask, (num_subjects, 1, 1))

    # Transform raw data.
    binary_fc_matrices = np.where(mask, 1, 0)
    real_fc_matrices = np.where(mask, raw_fc_matrices, 0)

    return binary_fc_matrices, real_fc_matrices


def get_data_thresholded_by_sample_values(
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


def get_data_thresholded_by_explicit_matrix(
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


def get_data_thresholded_by_random_matrix(
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
            2 * rng.random((num_subjects, num_regions, num_regions)) - 1
        )
    else:
        thresholding_matrix = 2 * rng.random((1, num_regions, num_regions)) - 1
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
        assert (
            type(threshold) == float
        ), f"Used {type(threshold)} instead of `float` for {DataThresholdingType.FIXED_THRESHOLD}."
        mask = np.where(thresholding_operator(fc, threshold), True, False)
    elif threshold_type == DataThresholdingType.KNN:
        assert (
            type(threshold) == int
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
        raise NotImplementedError()

    # Transform raw data.
    binary_fc_matrices = np.where(mask, 1, 0)
    real_fc_matrices = np.where(mask, raw_fc_matrices, 0)

    return binary_fc_matrices, real_fc_matrices
