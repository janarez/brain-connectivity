"""
Contains general training class for training any model.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data.dataloader import DataLoader
from torchinfo.torchinfo import summary

from .dataset import FunctionalConnectivityDataset
from .dense import ConnectivityDenseNet
from .evaluation import ModelEvaluation
from .general_utils import close_logger, get_logger
from .gin import GIN
from .model import Model

# Converts dictionary to string, keeps only first letters of words in keys.
stringify = lambda d: "_".join(
    [
        f"{''.join([w[0] for w in k.split('_')])}={v}"
        if type(v) != dict
        else stringify(v)
        for k, v in d.items()
    ]
)


model_param_names = [
    "num_hidden_features",
    "num_sublayers",
    "dropout",
]
dense_param_names = [
    "mode",
    "num_nodes",
    "readout",
    "emb_dropout",
    "emb_residual",
    "emb_init_weights",
    "emb_val",
    "emb_std",
]
graph_param_names = ["eps"]
dataset_param_names = [
    "upsample_ts",
    "upsample_ts_method",
    "correlation_type",
    "node_features",
    "batch_size",
    "geometric_kwargs",
]
training_param_names = ["epochs", "optimizer_kwargs"]


class Trainer:
    # TODO: Write class docstring.
    def __init__(
        self,
        log_folder,
        epochs: int,
        validation_frequency: int,
        criterion,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_kwargs: Optional[dict] = None,
        fc_matrix_plot_frequency: Optional[int] = None,
        fc_matrix_plot_sublayer: int = 0,
    ):
        self.logger = get_logger(
            "trainer", os.path.join(log_folder, "trainer.txt")
        )
        self.log_folder = log_folder
        self.evaluation = ModelEvaluation(log_folder)

        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._scheduler = scheduler
        self._scheduler_kwargs = (
            scheduler_kwargs if scheduler_kwargs is not None else {}
        )

        self.criterion = criterion

        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.train_loss, self.eval_loss = [], []

        self.fc_matrix_plot_frequency = fc_matrix_plot_frequency
        self.fc_matrix_plot_sublayer = fc_matrix_plot_sublayer

        # Log training setting.
        self.logger.debug(f"Optimizer: {optimizer}, {optimizer_kwargs}")
        self.logger.debug(f"Scheduler: {scheduler} {scheduler_kwargs}")
        self.logger.debug(f"Criterion: {self.criterion}")
        self.logger.debug(f"Epochs: {self.epochs}")
        self.logger.debug(f"Validation frequency: {self.validation_frequency}")
        self.logger.debug(
            f"Plot FC matrix: freq {self.fc_matrix_plot_frequency}, sublayer {self.fc_matrix_plot_sublayer}"
        )

    def train(
        self,
        model: Model,
        named_trainloader: Tuple[str, DataLoader],
        named_evalloader: Tuple[str, DataLoader],
        fold: int,
    ):
        train_dataset, trainloader = named_trainloader
        eval_dataset, evalloader = named_evalloader
        self.train_loss.append([])
        self.eval_loss.append([])

        "Runs training, all relevant arguments must be provided on class creation."
        self.optimizer = self._optimizer(
            model.parameters(), **self._optimizer_kwargs
        )
        self.scheduler = (
            self._scheduler(self.optimizer, **self._scheduler_kwargs)
            if self._scheduler is not None
            else None
        )
        self.evaluation.set_fold(fold)
        self.writer = SummaryWriter(os.path.join(self.log_folder, str(fold)))

        for epoch in range(self.epochs):
            # Train epoch.
            evaluate = (epoch + 1) % self.validation_frequency == 0
            loss = self._epoch_step(
                model,
                trainloader,
                epoch=epoch,
                dataset=train_dataset,
                evaluate=evaluate,
            )
            self.train_loss[fold].append(loss)
            self.logger.debug(f"Epoch {epoch}: {train_dataset} loss = {loss}")

            if evaluate:
                self.evaluation.log_evaluation(
                    epoch, train_dataset, self.writer
                )

            # Evaluate epoch.
            if (epoch + 1) % self.validation_frequency == 0:
                loss = self._epoch_step(
                    model,
                    evalloader,
                    epoch=epoch,
                    dataset=eval_dataset,
                    evaluate=True,
                )
                # FIXME: Will currently work only with `validation_freq == 1`.
                self.eval_loss[fold].append(loss)
                self.logger.debug(
                    f"Epoch {epoch}: {eval_dataset} loss = {loss}"
                )
                self.evaluation.log_evaluation(epoch, eval_dataset, self.writer)

            # Plot connectivity matrix.
            if (
                self.fc_matrix_plot_frequency is not None
                and (epoch + 1) % self.fc_matrix_plot_frequency == 0
            ):
                model.plot_fc_matrix(
                    epoch, sublayer=self.fc_matrix_plot_sublayer
                )

    def get_results(self, train_dataset, eval_dataset):
        train_res = self.evaluation.get_experiment_results(train_dataset)
        eval_res = self.evaluation.get_experiment_results(eval_dataset)
        train_res["loss"] = (
            np.mean(self.train_loss, axis=0),
            np.std(self.train_loss, axis=0),
        )
        eval_res["loss"] = (
            np.mean(self.eval_loss, axis=0),
            np.std(self.eval_loss, axis=0),
        )

        # Close all loggers used in single inner CV run.
        for l in ["dataset", "evaluation", "trainer"]:
            close_logger(l)
        return train_res, eval_res

    def _epoch_step(
        self,
        model: Model,
        dataloader: DataLoader,
        epoch: int,
        dataset: str,
        evaluate: bool,
    ):
        backpropagate = dataset in ["train", "dev"]
        running_loss = 0.0
        if backpropagate:
            model.train()
        else:
            model.eval()

        for data in dataloader:
            loss, outputs = self._model_step(
                model, data, backpropagate=backpropagate
            )
            running_loss += loss

            # Calculate evaluation metrics.
            if evaluate:
                self.evaluation.evaluate(outputs.view(-1), data.y.view(-1))

        epoch_loss = running_loss / len(dataloader)

        # Update learning rate.
        if self.scheduler is not None and not backpropagate:
            self.scheduler.step(epoch_loss)
            self.logger.debug(
                f"Epoch {epoch}: learning rate = {self.optimizer.param_groups[0]['lr']}"
            )

        self.writer.add_scalar(f"{dataset} loss", epoch_loss, epoch)
        return epoch_loss

    def _model_step(self, model, data, backpropagate: bool = True):
        self.optimizer.zero_grad()
        outputs = model(data)

        loss = self.criterion(outputs.view(-1), data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs


def init_traning(
    model_type,
    log_folder,
    data_folder,
    device,
    hyperparameters,
    targets,
    model_params,
    training_params,
):
    # Prepare model.
    model_class = GIN if model_type == "graph" else ConnectivityDenseNet
    specific_model_param_names = (
        graph_param_names if model_type == "graph" else dense_param_names
    )
    model_arguments = {
        **model_params,
        **{k: hyperparameters[k] for k in model_param_names},
        **{k: hyperparameters[k] for k in specific_model_param_names},
    }
    model_class.log(log_folder, model_arguments)

    data = FunctionalConnectivityDataset(
        targets=targets,
        data_folder=data_folder,
        device=device,
        **{k: hyperparameters[k] for k in dataset_param_names},
        log_folder=log_folder,
    )

    trainer = Trainer(
        **training_params,
        **{k: hyperparameters[k] for k in training_param_names},
        log_folder=log_folder,
    )

    return model_class, model_arguments, data, trainer
