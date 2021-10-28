import logging
import os
import pickle
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, DataLoader

from .data_utils import DenseDataset, dotdict, identity_matrix, zeroth_axis_sample
from .enums import CorrelationType, NodeFeatures
from .fc_matrix import create_connectivity_matrices


class FunctionalConnectivityDataset:
    def __init__(
        self,
        log_folder,
        folder,
        dataframe_with_subjects: str = "patients-cleaned.csv",
        target_column: str = "target",
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        node_features: NodeFeatures = NodeFeatures.FC_MATRIX_ROW,
        test_size: Union[int, float] = 50,
        batch_size: int = 8,
        num_folds: int = 7,
        random_seed: int = 42,
        geometric_dataset_kwargs: dict = {},
    ):
        self.log_folder = log_folder
        self.base_folder = folder

        df = pd.read_csv(os.path.join(self.base_folder, dataframe_with_subjects), index_col=0)
        self.raw_fc_matrices = self._read_raw_matrices(correlation_type)
        self.targets = df[target_column].values

        self.num_subjects, self.num_regions, _ = self.raw_fc_matrices.shape
        self.batch_size = batch_size
        self.skf = StratifiedKFold(n_splits=num_folds, random_state=random_seed, shuffle=True)

        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Parameters for setting up node features.
        self.node_features = node_features
        self.geometric_dataset_kwargs = geometric_dataset_kwargs

        # Split data into hidden test set and dev (train+validation) set.
        self.dev_indices, self.test_indices = self._dev_test_split(test_size=test_size)
        self.train_indices, self.val_indices = self.next_fold()

        # Save to file before training.
        self._log()

    def _log(self):
        """
        Logs all important information about used dataset to a file.
        """
        with open(os.path.join(self.log_folder, "dataset.txt"), "w", encoding="utf-8") as f:
            f.write(self.__dict__.__str__())

    def _read_raw_matrices(self, correlation_type: CorrelationType):
        path = os.path.join(self.base_folder, f"fc-{correlation_type}.pickle")
        with open(path, "rb") as f_rb:
            f = pickle.load(f_rb)

        logging.debug(f"Loaded dataset with shape {f.shape}")
        return f

    def _dev_test_split(self, test_size):
        patient_indexes = np.argwhere(self.targets == 1).reshape(-1)
        control_indexes = np.argwhere(self.targets == 0).reshape(-1)

        if type(test_size) == float:
            test_size = round(self.num_subjects * test_size)

        test_patients = self.rng.choice(patient_indexes, size=test_size // 2, replace=False)
        test_controls = self.rng.choice(control_indexes, size=test_size // 2, replace=False)

        test_indices = np.hstack([test_patients, test_controls])
        train_indices = np.array(list(set(range(self.num_subjects)) - set(test_indices)))

        assert len(test_indices) + len(train_indices) == self.num_subjects, "Dataset splitting created / lost samples."
        logging.debug(f"Test size: {len(test_indices)}")
        logging.debug(f"Test group 1 percantage: {sum(test_indices) / len(test_indices) * 100:.2f}")
        logging.debug(f"Train size: {len(train_indices)}")
        logging.debug(f"Train group 1 percantage: {sum(train_indices) / len(train_indices) * 100:.2f}")

        return train_indices, test_indices

    def _get_node_features_function(self):
        # Each node contains its row of the raw correlation matrix.
        if self.node_features == NodeFeatures.FC_MATRIX_ROW:
            node_features_function = partial(zeroth_axis_sample, self.raw_fc_matrices)
        # Each node contains one hot encoding of its brain region id.
        elif self.node_features == NodeFeatures.ONE_HOT_REGION:
            node_features_function = partial(identity_matrix, self.num_regions)
        else:
            raise ValueError(f"Unknown value of `node_features` - ({self.node_features}). Use the `NodeFeatures` enum.")
        return node_features_function

    def _get_dense_dataset(self, indices):
        node_features_function = self._get_node_features_function()
        return DenseDataset(
            node_features_function(indices).to(torch.float32),
            torch.from_numpy(self.targets[indices]).to(torch.int64),
        )

    def _get_geometric_dataset(self, indices):
        """
        `Data` object fields

        - `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
        - `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
        - `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
        - `data.y`: Target to train against (may have arbitrary shape), e.g., node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`
        - `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`
        """
        # We create FC data every time, since the average method needs to be calculated on training data only (i.e. before each fold)
        # and the overhead is small.
        binary_fc_matrices, real_fc_matrices = create_connectivity_matrices(
            self.raw_fc_matrices,
            **self.geometric_dataset_kwargs,
            binary_targets=self.targets,
            # Only training info, no val / test data.
            train_indices=self.train_indices,
        )
        node_features_function = self._get_node_features_function()
        return [
            Data(
                x=node_features_function([i]).squeeze(0),
                edge_index=torch.from_numpy(np.asarray(np.nonzero(binary_fc_matrices[i]))).to(torch.int64),
                y=torch.tensor([target], dtype=torch.int64),
            )
            for target, i in zip(self.targets[indices], indices)
        ]

    @property
    def geometric_trainloader(self):
        return DataLoader(self._get_geometric_dataset(self.train_indices), batch_size=self.batch_size, shuffle=True)

    @property
    def geometric_valloader(self):
        return DataLoader(self._get_geometric_dataset(self.val_indices), batch_size=self.batch_size, shuffle=False)

    @property
    def geometric_testloader(self):
        return DataLoader(self._get_geometric_dataset(self.test_indices), batch_size=self.batch_size, shuffle=False)

    @property
    def dense_trainloader(self):
        return list(
            map(
                dotdict,
                DataLoader(self._get_dense_dataset(self.train_indices), batch_size=self.batch_size, shuffle=True),
            )
        )

    @property
    def dense_valloader(self):
        # We need to map to `dotdict` here, because `DataLoader` always returns plain dict from batch collating.
        return list(
            map(
                dotdict,
                DataLoader(self._get_dense_dataset(self.val_indices), batch_size=self.batch_size, shuffle=False),
            )
        )

    @property
    def dense_testloader(self):
        return list(
            map(
                dotdict,
                DataLoader(self._get_dense_dataset(self.test_indices), batch_size=self.batch_size, shuffle=False),
            )
        )

    def next_fold(self):
        "Updates train and validation indices to that of next stratified fold."
        train_split, val_split = next(self.skf.split(self.dev_indices, self.targets[self.dev_indices]))
        return self.dev_indices[train_split], self.dev_indices[val_split]
