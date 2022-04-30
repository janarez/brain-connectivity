import os
import pickle
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch_geometric

from .data_utils import (
    DenseDataset,
    aaft_surrogates,
    calculate_correlation_matrix,
    dotdict_collate,
    iaaft_surrogates,
    identity_matrix,
    identity_matrix_concat_zeroth_axis_sample,
    one,
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

    hyperparameters = [
        "upsample_ts",
        "upsample_ts_method",
        "correlation_type",
        "node_features",
        "batch_size",
        "graph_kwargs",
    ]

    def __init__(
        self,
        log_folder,
        data_folder,
        device,
        targets: np.array,
        upsample_ts: Optional[int] = None,
        upsample_ts_method: Optional[str] = None,
        correlation_type: CorrelationType = CorrelationType.PEARSON,
        node_features: NodeFeatures = NodeFeatures.FC_MATRIX_ROW,
        batch_size: int = 8,
        graph_kwargs: Optional[dict] = None,
    ):
        self.logger = get_logger(
            "dataset", os.path.join(log_folder, "dataset.txt")
        )
        self.device = device

        self.raw_fc_matrices, self.raw_fc_surrogates = self._get_raw_matrices(
            correlation_type, upsample_ts, upsample_ts_method, data_folder
        )
        self.targets = targets

        self.num_subjects, self.num_regions, _ = self.raw_fc_matrices.shape
        self.batch_size = batch_size

        # Parameters for setting up node features.
        self.node_features = node_features
        self.graph_kwargs = graph_kwargs

        self.logger.debug(f"Upsample timeseries: {upsample_ts}")
        self.logger.debug(f"Upsample timeseries method: {upsample_ts_method}")
        self.logger.debug(f"Correlation: {correlation_type}")
        self.logger.debug(f"Node features: {node_features}")
        self.logger.debug(f"Batch size: {batch_size}")
        self.logger.debug(f"Graph kwargs: {graph_kwargs}")

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
            self.logger.debug(f"Loaded {path} from cache.")
            # Return early if not upsampling.
            if upsample_ts is None:
                return raw_matrices, None

        # Optionally check cache for upsampled data.
        if upsample_ts is not None and os.path.exists(surr_path):
            with open(surr_path, "rb") as f:
                raw_surrogates = pickle.load(f)
            self.logger.debug(f"Loaded {surr_path} from cache.")
            # Return if all data has been loaded.
            if raw_matrices is not None:
                return raw_matrices, raw_surrogates

        # Otherwise compute from raw timeseries data.
        with open(os.path.join(data_folder, "timeseries.pickle"), "rb") as f:
            ts = pickle.load(f)
            self.logger.info(
                f"Loaded raw timeseries dataset with shape {ts.shape}"
            )

        # Not even non-upsampled data were cached.
        if raw_matrices is None:
            raw_matrices = calculate_correlation_matrix(ts, correlation_type)
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
            raw_surrogates = calculate_correlation_matrix(
                surrogates.reshape(-1, *surrogates.shape[-2:]), correlation_type
            ).reshape(*surrogates.shape[:-1], -1)
            # Cache.
            with open(surr_path, "wb") as f:
                pickle.dump(raw_surrogates, f)

        return raw_matrices, raw_surrogates

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
        # Each node contains a `num_regions` ones.
        elif self.node_features == NodeFeatures.ONE:
            node_features_function = partial(one, self.num_regions)
        # Each node contains concat of one hot and fc row.
        elif self.node_features == NodeFeatures.ONE_HOT_CAT_FC_ROW:
            node_features_function = partial(
                identity_matrix_concat_zeroth_axis_sample,
                self.raw_fc_matrices if not sur else self.raw_fc_surrogates,
            )
        else:
            raise ValueError(
                f"Unknown value of `node_features` - ({self.node_features}). Use the `NodeFeatures` enum."
            )
        return node_features_function

    def _get_dense_dataset(self, indices, view):
        node_features_function = self._get_node_features_function()
        X = self._get_dense_view(node_features_function(indices), view=view)
        y = torch.from_numpy(self.targets[indices])
        orig_dataset = DenseDataset(
            X.to(torch.float32).to(self.device),
            y.to(torch.float32).to(self.device),
        )

        # Append surrogate data.
        if self.raw_fc_surrogates is not None:
            node_features_function = self._get_node_features_function(sur=True)
            X = self._get_dense_view(
                node_features_function(indices)
                # Merge index and upsample dims into one.
                .view(-1, self.num_regions, self.num_regions),
                view=view,
            )
            y = torch.from_numpy(self.targets[indices]).repeat_interleave(
                # Labels must be repeated `upsample` times.
                self.raw_fc_surrogates.shape[1]
            )

            sur_dataset = DenseDataset(
                X.to(torch.float32).to(self.device),
                y.to(torch.float32).to(self.device),
            )
            return orig_dataset + sur_dataset

        return orig_dataset

    def _get_dense_view(self, x, view):
        # Flattens full matrix.
        if view == "flattened-dense":
            x = x.reshape((-1, x.shape[-1] ** 2))
        # Flattens only upper triangle without diagonal.
        elif view == "triangular-dense":
            x = torch.tensor(
                np.array(
                    [
                        np.hstack(
                            [
                                row[i + 1 :]  # noqa E203
                                for i, row in enumerate(sample)
                            ]
                        )
                        for sample in x
                    ]
                )
            )
        return x

    def _get_graph_dataset(self, indices):
        """
        `Data` object fields

        - `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`
        - `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`
        - `data.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]`
        - `data.y`: Graph-level targets of shape `[1, *]`
        - `data.pos`: Node position matrix with shape `[num_nodes, num_dimensions]`
        """
        binary_fc_matrices, _ = create_connectivity_matrices(
            self.raw_fc_matrices[indices],
            **self.graph_kwargs,
        )
        node_features_function = self._get_node_features_function()
        orig_dataset = [
            torch_geometric.data.Data(
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
                    [self.targets[i]], dtype=torch.float32, device=self.device
                ),
            ).to(self.device)
            for i, fc in zip(indices, binary_fc_matrices)
        ]

        if self.raw_fc_surrogates is not None:
            binary_fc_surrogates, _ = create_connectivity_matrices(
                self.raw_fc_surrogates[indices].reshape(
                    -1, self.num_regions, self.num_regions
                ),
                **self.graph_kwargs,
            ).reshape(len(indices), -1, self.num_regions, self.num_regions)

            node_features_function = self._get_node_features_function(sur=True)
            sur_dataset = [
                torch_geometric.data.Data(
                    x=x.to(torch.float32).to(self.device),
                    edge_index=torch.from_numpy(np.asarray(np.nonzero(fc)))
                    .to(torch.int64)
                    .to(self.device),
                    y=torch.tensor(
                        [self.targets[i]],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ).to(self.device)
                for i in indices
                for x, fc in zip(
                    node_features_function([i]), binary_fc_surrogates[i]
                )
            ]
            return orig_dataset + sur_dataset

        return orig_dataset

    def dataloader(self, dataset, indices, view):
        self._log_loader_stats(dataset, indices)
        if view == "graph":
            return self._graph_loader(dataset, indices)
        else:
            return self._dense_loader(dataset, indices, view)

    def _graph_loader(self, dataset, indices):
        "Dataloader with data for graph neural network."
        # NOTE: Trainloader needs to be access before val and test loader to set avg diff matrix properly.
        if (
            dataset in ["train", "dev"]
            and self.graph_kwargs["thresholding_function"]
            == ThresholdingFunction.GROUP_AVERAGE
        ):
            self.graph_kwargs[
                "thresholding_matrix"
            ] = get_matrix_of_avg_diff_between_groups(
                self.raw_fc_matrices, self.targets, indices
            )
        return torch_geometric.loader.DataLoader(
            self._get_graph_dataset(indices),
            batch_size=self.batch_size,
            shuffle=dataset in ["train", "dev"],
        )

    def _dense_loader(self, dataset, indices, view):
        "Dataloader with data for dense neural network."
        self._log_loader_stats(dataset, indices)
        return torch.utils.data.dataloader.DataLoader(
            self._get_dense_dataset(indices, view=view),
            batch_size=self.batch_size,
            shuffle=dataset in ["train", "dev"],
            collate_fn=dotdict_collate,
        )

    def ml_loader(self, dataset, indices, flatten):
        "Data for standard machine learning algorithms as X, y matrices."
        self._log_loader_stats(dataset, indices)
        X = self._get_dense_view(self.raw_fc_matrices[indices], view=flatten)
        y = self.targets[indices]
        return X, y

    def _log_loader_stats(self, dataset, indices):
        dataset = dataset.title()
        self.logger.debug(f"{dataset} size: {len(indices)}")
        self.logger.debug(f"{dataset} indices: {indices}")
        self.logger.debug(
            f"{dataset} 1:0: {sum(self.targets[indices]) / len(indices)}"
        )
