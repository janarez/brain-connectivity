from functools import partial
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional, Union
import operator

import torch
from torch.utils.data import TensorDataset
from torch_geometric.data import Data, DataLoader

from .enums import CorrelationType, NodeFeatures, ThresholdingFunction, DataThresholdingType
from .fc_matrix import get_data_threshold_at_largest_average_difference_between_groups, \
    get_data_thresholded_by_explicit_matrix, get_data_thresholded_by_random_matrix, \
    get_data_thresholded_by_sample_values
from .data_utils import dotdict, identity_matrix, zeroth_axis_sample


class FunctionalConnectivityDataset:

    def __init__(
        self,
        log_folder,
        folder: os.Path,
        dataframe_with_subjects: str = 'patients_cleaned.csv',
        target_column: str = 'target',
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        node_features: NodeFeatures = NodeFeatures.FC_MATRIX_ROW,
        test_size: Union[int, float] = 50,
        batch_size: int = 8,
        random_seed: int = 42
    ):
        self.log_folder = log_folder
        self.base_folder = folder
        self.df = pd.read_csv(os.path.join(
            self.base_folder, dataframe_with_subjects), index_col=0)

        self.raw_fc_matrices = self._read_raw_matrices(correlation_type)
        self.targets = self.df[target_column].values

        self.num_subjects, self.num_regions, _ = self.raw_fc_matrices.shape
        self.batch_size = batch_size

        self.random_seed = random_seed
        self.rgn = np.random.default_rng(random_seed)

        # Select how node features are obtained.
        if node_features == NodeFeatures.FC_MATRIX_ROW:
            self.get_node_features = partial(
                zeroth_axis_sample, self.raw_fc_matrices)
        elif node_features == NodeFeatures.ONE_HOT_REGION:
            self.get_node_features = partial(
                identity_matrix, self.num_regions)

        # Split data into hidden test set and training (dev+train) set.
        self.train_indices, self.test_indices = self._train_test_split(
            test_size=test_size)

        # Save to file before training.
        self._log()

    def _log(self):
        """
        Logs all important information about used dataset to a file.
        """
        with open(os.path.join(self.log_folder, 'dataset.txt'), 'w', encoding="utf-8") as f:
            f.write(self.__dict__.__str__())

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

    def _train_test_split(self, test_size):
        patient_indexes = self.df[self.df[self.target_column] == 1].index
        control_indexes = self.df[self.df[self.target_column] == 0].index

        if type(test_size) == float:
            test_size = round(self.num_subjects * test_size)

        test_patients = self.rng.choice(
            patient_indexes, size=test_size//2, replace=False)
        test_controls = self.rng.choice(
            control_indexes, size=test_size//2, replace=False)

        test_indices = np.hstack([test_patients, test_controls])
        train_indices = list(
            set(range(self.num_subjects)) - set(test_indices))

        assert len(test_indices) + len(
            train_indices) == self.num_subjects, 'Dataset splitting created / lost samples.'
        logging.debug(f'Test size: {len(test_indices)}')
        logging.debug(
            f'Test group 1 percantage: {sum(test_indices) / len(test_indices) * 100:.2f}')
        logging.debug(f'Train size: {len(train_indices)}')
        logging.debug(
            f'Train group 1 percantage: {sum(train_indices) / len(train_indices) * 100:.2f}')

        return train_indices, test_indices

    def _get_dense_dataset(self, train_set: bool):
        indices = self.train_indices if train_set else self.test_indices
        return TensorDataset(
            torch.from_numpy(self.raw_fc_matrices[indices]).to(torch.float32),
            torch.from_numpy(self.targets[indices]).to(torch.int64)
        )

    def _get_geometric_dataset(self, train_set: bool):
        """
        `Data` object fields

        - `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
        - `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
        - `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
        - `data.y`: Target to train against (may have arbitrary shape), e.g., node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`
        - `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`
        """
        indices = self.train_indices if train_set else self.test_indices
        return [Data(
            x=self.get_node_features(i),
            edge_index=torch.from_numpy(np.asarray(
                np.nonzero(self.binary_fc_matrices[i]))).to(torch.int64),
            y=torch.tensor([target], dtype=torch.int64)
        ) for target, i in zip(self.targets[indices], indices)
        ]

    @property
    def geometric_trainloader(self):
        pass

    @property
    def geometric_valloader(self):
        pass

    @property
    def geometric_testloader(self):
        return DataLoader(self._get_geometric_dataset(train_set=False), batch_size=self.batch_size, shuffle=False)

    @property
    def dense_trainloader(self):
        pass

    @property
    def dense_valloader(self):
        pass

    @property
    def dense_testloader(self):
        return list(map(dotdict, DataLoader(self._get_dense_dataset(train_set=False), batch_size=self.batch_size, shuffle=False)))
