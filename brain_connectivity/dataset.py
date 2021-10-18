import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional, Union
import operator

from .enums import CorrelationType, ThresholdingFunction, DataThresholdingType
from .fc_matrix import get_data_threshold_at_largest_average_difference_between_groups, \
    get_data_thresholded_by_explicit_matrix, get_data_thresholded_by_random_matrix, \
    get_data_thresholded_by_sample_values

class FunctionalConnectivityDataset:

    def __init__(
        self,
        folder: os.Path,
        dataframe_with_subjects: str = 'patients_cleaned.csv',
        target_column: str = 'target',
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        random_seed: int = 42
    ):
        self.base_folder = folder
        self.df =  pd.read_csv(os.path.join(self.base_folder, dataframe_with_subjects), index_col=0)
        self.raw_fc_matrices = self._read_raw_matrices(correlation_type)
        self.num_subjects, self.num_regions, _ = self.raw_fc_matrices.shape

        self.random_seed = random_seed
        self.rgn = np.random.default_rng(random_seed)


    def _read_raw_matrices(self, correlation_type: CorrelationType):
        path = os.path.join(self.base_folder, f'fc-{correlation_type}.pickle')
        with open(path, 'rb') as f_rb:
            f = pickle.load(f_rb)

        logging.debug(f'Loaded dataset with shape {f.shape}')
        return f


    def create_connectivity_dataset(
        self,
        thresholding_function: ThresholdingFunction,
        threshold_type: DataThresholdingType,
        threshold: Union[float, int],
        thresholding_operator: Optional[Union[operator.le, operator.ge]] = operator.ge,
        threshold_by_absolute_value: bool = True,
        return_absolute_value: bool = False
    ):
        if thresholding_function == ThresholdingFunction.GROUP_AVERAGE:
            b, r = get_data_threshold_at_largest_average_difference_between_groups(
                raw_fc_matrices=self.raw_fc_matrices,
                binary_targets=None,
                train_indices=None,
                threshold_type=threshold_type,
                threshold=threshold,
                thresholding_operator=thresholding_operator
            )
        elif thresholding_function == ThresholdingFunction.SUBJECT_VALUES:
            b, r = get_data_thresholded_by_sample_values(
                raw_fc_matrices=self.raw_fc_matrices,
                threshold_type=threshold_type,
                threshold=threshold,
                thresholding_operator=thresholding_operator,
                threshold_by_absolute_value=threshold_by_absolute_value,
                return_absolute_value=return_absolute_value
            )
        elif thresholding_function == ThresholdingFunction.EXPLICIT_MATRIX:
            b, r = get_data_thresholded_by_explicit_matrix(
                raw_fc_matrices=self.raw_fc_matrices,
                thresholding_matrix=None,
                threshold_type=threshold_type,
                threshold=threshold,
                thresholding_operator=thresholding_operator,
                threshold_by_absolute_value=threshold_by_absolute_value,
                return_absolute_value=return_absolute_value
            )
        elif thresholding_function == ThresholdingFunction.RANDOM:
            b, r = get_data_thresholded_by_random_matrix(
                raw_fc_matrices=self.raw_fc_matrices,
                per_subject=None,
                threshold_type=threshold_type,
                threshold=threshold,
                thresholding_operator=thresholding_operator,
                threshold_by_absolute_value=threshold_by_absolute_value,
                return_absolute_value=return_absolute_value
            )

        self.binary_fc_matrices = b
        self.real_fc_matrices = r


    def train_test_split(self, test_size: int = 50):
        patient_indexes = self.df[self.df[self.target_column] == 1].index
        control_indexes = self.df[self.df[self.target_column] == 0].index

        test_patients = self.rng.choice(patient_indexes, size=test_size//2, replace=False)
        test_controls = self.rng.choice(control_indexes, size=test_size//2, replace=False)

        self.test_indices = np.hstack([test_patients,test_controls])
        self.train_indices = list(set(range(self.num_subjects)) - set(self.test_indices))

        assert len(self.test_indices) + len(self.train_indices) == self.num_subjects, 'Dataset splitting created / lost samples.'
        logging.debug(f'Test size: {len(self.test_indices)}')
        logging.debug(f'Test group 1 percantage: {sum(self.test_indices) / len(self.test_indices) * 100:.2f}')
        logging.debug(f'Train size: {len(self.train_indices)}')
        logging.debug(f'Train group 1 percantage: {sum(self.train_indices) / len(self.train_indices) * 100:.2f}')

        return self.train_indices, self.test_indices
