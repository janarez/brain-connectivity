import os
import pickle
from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import StratifiedKFold

from .data_utils import (
    DenseDataset,
    aaft_surrogates,
    dotdict_collate,
    iaaft_surrogates,
    identity_matrix,
    zeroth_axis_sample,
)
from .enums import CorrelationType, NodeFeatures, ThresholdingFunction
from .fc_matrix import (
    create_connectivity_matrices,
    get_matrix_of_avg_diff_between_groups,
)
from .general_utils import get_logger


class FunctionalConnectivityDataset:
    """
    TODO: Write class docstring.
    """

    def __init__(
        self,
        log_folder,
        data_folder,
        device,
        dataframe_with_subjects: str = "patients-cleaned.csv",
        target_column: str = "target",
        upsample_ts: Optional[int] = None,
        upsample_ts_method: Optional[str] = None,
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        node_features: NodeFeatures = NodeFeatures.FC_MATRIX_ROW,
        batch_size: int = 8,
        num_folds: int = 7,
        random_seed: int = 42,
        geometric_kwargs: Optional[dict] = None,
    ):
        try:
            os.makedirs(log_folder, exist_ok=False)
        except FileExistsError as e:
            raise ValueError(
                f"Run experiment with same NAME and ID ({log_folder})."
            ) from e
        self.logger = get_logger(
            "dataset", os.path.join(log_folder, "dataset.txt")
        )
        self.device = device

        df = pd.read_csv(
            os.path.join(data_folder, dataframe_with_subjects), index_col=0
        )
        self.raw_fc_matrices, self.raw_fc_surrogates = self._get_raw_matrices(
            correlation_type, upsample_ts, upsample_ts_method, data_folder
        )
        self.targets = df[target_column].values

        self.num_subjects, self.num_regions, _ = self.raw_fc_matrices.shape
        self.batch_size = batch_size

        self.rng = np.random.default_rng(random_seed)

        # Parameters for setting up node features.
        self.node_features = node_features
        self.geometric_kwargs = geometric_kwargs

        # Stratified spliting of dataset.
        self.num_folds = num_folds
        self.skf = StratifiedKFold(
            n_splits=num_folds, random_state=random_seed, shuffle=True
        )
        # Iterator over outter CV: assessment of model performance.
        self.outer_cv_iterator = enumerate(
            self.skf.split(np.empty(shape=self.num_subjects), self.targets)
        )

    def _get_raw_matrices(
        self, correlation_type, upsample_ts, upsample_ts_method, data_folder
    ):
        # Check for cached pickles.
        path = os.path.join(
            data_folder, "cache", f"raw_matrices_{correlation_type}.pickle"
        )
        surr_path = (
            None
            if upsample_ts is None
            else os.path.join(
                data_folder,
                "cache",
                f"raw_surrogates_{correlation_type}_{upsample_ts}_{upsample_ts_method}.pickle",
            )
        )
        raw_matrices = None
        raw_surrogates = None

        # Check cache for original data.
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw_matrices = pickle.load(f)
            self.logger.info(f"Loaded {path} from cache.")
            # Return early if not upsampling.
            if upsample_ts is None:
                return raw_matrices, None

        # Optionally check cache for upsampled data.
        if upsample_ts is not None and os.path.exists(surr_path):
            with open(surr_path, "rb") as f:
                raw_surrogates = pickle.load(f)
            self.logger.info(f"Loaded {surr_path} from cache.")
            # Return if all data has been loadede.
            if raw_matrices is None:
                return raw_matrices, raw_surrogates

        # Otherwise compute from raw timeseries data.
        with open(os.path.join(data_folder, f"timeseries.pickle"), "rb") as f:
            ts = pickle.load(f)
            self.logger.info(
                f"Loaded raw timeseries dataset with shape {ts.shape}"
            )

        # Not even non-upsampled data were cached.
        if raw_matrices is None:
            raw_matrices = self._calculate_correlation_matrix(
                correlation_type, ts
            )
            # Cache.
            with open(path, "wb") as f:
                pickle.dump(raw_matrices, f)
        # Optionally upsample timeseries.
        if upsample_ts is not None:
            if upsample_ts_method == "aaft":
                surrogates = aaft_surrogates(ts, upsample=upsample_ts)
            elif upsample_ts_method == "iaaft":
                surrogates = iaaft_surrogates(ts, upsample=upsample_ts)
            else:
                raise ValueError(
                    f"Got {upsample_ts_method} and {upsample_ts}, accepting 'aaft' or 'iaaft', plus positive int."
                )
            raw_surrogates = self._calculate_correlation_matrix(
                correlation_type, surrogates
            )
            # Cache.
            with open(surr_path, "wb") as f:
                pickle.dump(raw_surrogates, f)

        return raw_matrices, raw_surrogates

    def _calculate_correlation_matrix(self, correlation_type, timeseries):
        # Placeholder correlation matrices.
        num_subjects, num_regions, _ = timeseries.shape
        raw_matrices = np.empty((num_subjects, num_regions, num_regions))
        for i, ts in enumerate(timeseries):
            # TODO: Support partial corr via pinqouin's `pcorr`.
            raw_matrices[i] = pd.DataFrame(ts).T.corr(
                method=correlation_type.value
            )
        return raw_matrices

    def _get_node_features_function(self, sur=False):
        """
        Returns partial function that get features for nodes when passed list of their indices.

        Args:
            sur: Pass `True` if you need node features for surrogates.
        """
        # Each node contains its row of the raw correlation matrix.
        if self.node_features == NodeFeatures.FC_MATRIX_ROW:
            node_features_function = partial(
                zeroth_axis_sample,
                self.raw_fc_matrices if not sur else self.raw_fc_surrogates,
            )
        # Each node contains one hot encoding of its brain region id.
        elif self.node_features == NodeFeatures.ONE_HOT_REGION:
            node_features_function = partial(identity_matrix, self.num_regions)
        else:
            raise ValueError(
                f"Unknown value of `node_features` - ({self.node_features}). Use the `NodeFeatures` enum."
            )
        return node_features_function

    def _get_dense_dataset(self, indices):
        node_features_function = self._get_node_features_function()
        orig_dataset = DenseDataset(
            node_features_function(indices).to(torch.float32).to(self.device),
            torch.from_numpy(self.targets[indices])
            .to(torch.int64)
            .to(self.device),
        )

        # Append surrogate data.
        if self.raw_fc_surrogates is not None:
            node_features_function = self._get_node_features_function(sur=True)
            sur_dataset = DenseDataset(
                node_features_function(indices)
                # Merge index and upsample dims into one.
                .view(-1, self.num_regions, self.num_regions)
                .to(torch.float32)
                .to(self.device),
                torch.from_numpy(self.targets[indices])
                # Index dim corresponds to label, but must be repeated `upsample` times.
                .repeat(self.raw_fc_surrogates.shape[1], dim=1)
                .to(torch.int64)
                .to(self.device),
            )
            return orig_dataset + sur_dataset

        return orig_dataset

    def _get_geometric_dataset(self, indices):
        """
        `Data` object fields

        - `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
        - `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
        - `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
        - `data.y`: Target to train against (may have arbitrary shape), e.g., node-level targets of shape `[num_nodes, *]` or graph-level targets of shape `[1, *]`
        - `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`
        """
        binary_fc_matrices, _ = create_connectivity_matrices(
            self.raw_fc_matrices[indices],
            **self.geometric_kwargs,
        )
        node_features_function = self._get_node_features_function()
        orig_dataset = [
            torch_geometric.Data(
                # Remove the batch dim.
                x=node_features_function([i])
                .squeeze(0)
                .to(torch.float32)
                .to(self.device),
                # All nonzero coordinate pairs, x and y axis separately.
                edge_index=torch.from_numpy(np.asarray(np.nonzero(fc)))
                .to(torch.int64)
                .to(self.device),
                y=torch.tensor(
                    [self.targets[i]], dtype=torch.int64, device=self.device
                ),
            ).to(self.device)
            for i, fc in zip(indices, binary_fc_matrices)
        ]

        if self.raw_fc_surrogates is not None:
            binary_fc_surrogates, _ = create_connectivity_matrices(
                self.raw_fc_surrogates[indices].reshape(
                    -1, self.num_regions, self.num_regions
                ),
                **self.geometric_kwargs,
            ).reshape(len(indices), -1, self.num_regions, self.num_regions)

            node_features_function = self._get_node_features_function(sur=True)
            sur_dataset = [
                torch_geometric.Data(
                    x=x.to(torch.float32).to(self.device),
                    edge_index=torch.from_numpy(np.asarray(np.nonzero(fc)))
                    .to(torch.int64)
                    .to(self.device),
                    y=torch.tensor(
                        [self.targets[i]], dtype=torch.int64, device=self.device
                    ),
                ).to(self.device)
                for i in indices
                for x, fc in zip(
                    node_features_function([i]), binary_fc_surrogates[i]
                )
            ]
            return orig_dataset + sur_dataset

        return orig_dataset

    @property
    def geometric_trainloader(self):
        "Train dataloader with data for graph neural network."
        # FIXME: Trainloader needs to be access before val and test loader to set avg diff matrix properly.
        if (
            self.geometric_kwargs["thresholding_function"]
            == ThresholdingFunction.GROUP_AVERAGE
        ):
            self.geometric_kwargs[
                "thresholding_matrix"
            ] = get_matrix_of_avg_diff_between_groups(
                self.raw_fc_matrices, self.targets, self.train_indices
            )
        return torch_geometric.data.DataLoader(
            self._get_geometric_dataset(self.train_indices),
            batch_size=self.batch_size,
            shuffle=True,
        )

    @property
    def geometric_valloader(self):
        "Validation dataloader with data for graph neural network."
        return torch_geometric.data.DataLoader(
            self._get_geometric_dataset(self.val_indices),
            batch_size=self.batch_size,
            shuffle=False,
        )

    @property
    def geometric_testloader(self):
        "Test dataloader with data for graph neural network."
        return torch_geometric.data.DataLoader(
            self._get_geometric_dataset(self.test_indices),
            batch_size=self.batch_size,
            shuffle=False,
        )

    @property
    def dense_trainloader(self):
        "Train dataloader with data for dense neural network."
        return torch.utils.data.dataloader.DataLoader(
            self._get_dense_dataset(self.train_indices),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dotdict_collate,
        )

    @property
    def dense_valloader(self):
        "Validation dataloader with data for dense neural network."
        return torch.utils.data.dataloader.DataLoader(
            self._get_dense_dataset(self.val_indices),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dotdict_collate,
        )

    @property
    def dense_testloader(self):
        "Test dataloader with data for dense neural network."
        return torch.utils.data.dataloader.DataLoader(
            self._get_dense_dataset(self.test_indices),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dotdict_collate,
        )

    def next_outter_cv_fold(self):
        """
        Iterator function that updates train and validation indices to that of next stratified fold.
        If the fold generator is exhausted returns `False` else `True`.

        To be used in a `while` cycle.
        """
        try:
            i, (dev_split, test_split) = next(self.outer_cv_iterator)
            self.inner_cv_iterator = enumerate(
                self.skf.split(
                    np.empty(shape=len(dev_split)), self.targets[dev_split]
                )
            )
            self.dev_indices, self.test_indices = dev_split, test_split
        except StopIteration:
            return False

        self.logger.info(f"Generated outer fold {i+1}/{self.num_folds}")
        return True

    def next_inner_cv_fold(self):
        """
        Iterator function that updates train and validation indices to that of next stratified fold.
        If the fold generator is exhausted returns `False` else `True`.

        To be used in a `while` cycle.
        """
        try:
            i, (train_split, val_split) = next(self.inner_cv_iterator)
            self.train_indices, self.val_indices = (
                self.dev_indices[train_split],
                self.dev_indices[val_split],
            )
        except StopIteration:
            return False

        self.logger.info(f"Generated inner fold {i+1}/{self.num_folds}")
        return True
