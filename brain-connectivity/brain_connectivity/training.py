"""
Contains general training class for training models.
"""
import contextlib
import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader.dataloader import DataLoader
from tqdm import tqdm

from .evaluation import BinaryClassificationEvaluation, RegressionEvaluation
from .general_utils import close_logger, get_logger
from .models.model import Model


def stringify(d):
    """
    Converts dictionary `d` to string, keeps only first letters of words in keys.
    """
    return "_".join(
        [
            f"{''.join([w[0] for w in k.split('_')])}={v if not isinstance(v, Callable) else v.__name__}"
            if not isinstance(v, dict)
            else stringify(v)
            for k, v in d.items()
        ]
    )


def cosine_loss(preds, target):
    """
    Function that measures cosine loss between the target and input probabilities.
    Mean reduction.
    From paper: https://arxiv.org/pdf/1901.09054.pdf.
    """
    # We can do just dot product since both vectors are already normalized.
    cosine_dissimilarity = 1 - torch.sum(
        torch.vstack([1 - preds, preds]).T
        * torch.nn.functional.one_hot(target.long(), num_classes=2),
        axis=1,
    )
    return torch.mean(cosine_dissimilarity)


class Trainer:
    """
    Facilitates model training and evaluation.

    Args:
        log_folder (`str`): Path for daving logs.
        epochs (`int`): Number of training epochs.
        criterion (`Callable`): Loss function.
        optimizer (`torch.optim.Optimizer`): Optimizer class.
        optimizer_kwargs (dict): All settings for `optimizer`. Like learning rate or weight decay.
        scheduler (`Optional[torch.optim.lr_scheduler]`): Optional schedular class. Default `None`.
        scheduler_kwargs (`Optional[dict]`): All settings for `scheduler`. Default `None`.
        fc_matrix_plot_frequency (`Optional[int]`): How (if) often to plot FC matrix. Default `None`.
        fc_matrix_plot_sublayer (`int`): Layer from which to plot FC matrix. Default 0.
        binary_cls (`bool`): If `True` binary classification metrics are used else regression. Deafult `True`.
    """

    hyperparameters = [
        "epochs",
        "optimizer",
        "optimizer_kwargs",
        "criterion",
        "scheduler",
        "scheduler_kwargs",
    ]

    def __init__(
        self,
        log_folder,
        epochs: int,
        criterion,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_kwargs: Optional[dict] = None,
        fc_matrix_plot_frequency: Optional[int] = None,
        fc_matrix_plot_sublayer: int = 0,
        binary_cls: bool = True,
    ):
        self.logger = get_logger(
            "trainer", os.path.join(log_folder, "trainer.txt")
        )
        self.log_folder = log_folder
        eval_class = (
            BinaryClassificationEvaluation
            if binary_cls
            else RegressionEvaluation
        )
        self.evaluation = eval_class(log_folder)

        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._scheduler = scheduler
        self._scheduler_kwargs = (
            scheduler_kwargs if scheduler_kwargs is not None else {}
        )

        self.criterion = criterion

        self.epochs = epochs
        self.train_loss, self.eval_loss = [], []

        self.fc_matrix_plot_frequency = fc_matrix_plot_frequency
        self.fc_matrix_plot_sublayer = fc_matrix_plot_sublayer

        # Log training setting.
        self.logger.debug(f"Optimizer: {optimizer}, {optimizer_kwargs}")
        self.logger.debug(f"Scheduler: {scheduler} {scheduler_kwargs}")
        self.logger.debug(f"Criterion: {self.criterion}")
        self.logger.debug(f"Epochs: {self.epochs}")
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
        """
        Trains and evaluates model on given data.

        Args:
            model (`Model`): Instance of model.
            named_trainloader (`Tuple[str, DataLoader]`):`DataLoader` with evaluation data
                and dataset identifier from ["train", "dev"].
            named_evalloader (`Tuple[str, DataLoader]`): `DataLoader` with evaluation data
                and dataset identifier from ["val", "test"].
            fold (`int`): CV fold ID.
        """
        train_dataset, trainloader = named_trainloader
        eval_dataset, evalloader = named_evalloader
        self.train_loss.append([])
        self.eval_loss.append([])

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

        epoch_progress = tqdm(
            range(self.epochs), total=self.epochs, leave=False, desc="Training"
        )
        for epoch in epoch_progress:
            # Train epoch.
            loss = self._epoch_step(
                model,
                trainloader,
                epoch=epoch,
                dataset=train_dataset,
            )
            self.train_loss[fold].append(loss)
            self.logger.debug(f"Epoch {epoch}: {train_dataset} loss = {loss}")
            epoch_progress.set_postfix({"loss": loss})

            # Evaluate epoch.
            self.evaluation.log_evaluation(epoch, train_dataset, self.writer)
            loss = self._epoch_step(
                model,
                evalloader,
                epoch=epoch,
                dataset=eval_dataset,
            )
            self.eval_loss[fold].append(loss)
            self.logger.debug(f"Epoch {epoch}: {eval_dataset} loss = {loss}")
            self.evaluation.log_evaluation(epoch, eval_dataset, self.writer)

            # Plot connectivity matrix.
            if (
                self.fc_matrix_plot_frequency is not None
                and (epoch + 1) % self.fc_matrix_plot_frequency == 0
            ):
                model.plot_fc_matrix(
                    epoch,
                    sublayer=self.fc_matrix_plot_sublayer,
                    path=os.path.join(self.log_folder, str(fold)),
                )

    def get_results(self, train_dataset, eval_dataset):
        """
        Get metrics for given dataset identifiers ["train", "dev"] and  ["val", "test"].
        """
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
        for name in ["dataset", "evaluation", "trainer"]:
            close_logger(name)
        return train_res, eval_res

    def _epoch_step(
        self,
        model: Model,
        dataloader: DataLoader,
        epoch: int,
        dataset: str,
    ):
        backpropagate = dataset in ["train", "dev"]
        if backpropagate:
            model.train()
            cm = contextlib.nullcontext()
        else:
            model.eval()
            cm = torch.no_grad()

        with cm:
            running_loss = 0.0
            for data in dataloader:
                loss, outputs = self._model_step(
                    model, data, backpropagate=backpropagate
                )
                running_loss += loss

                # Calculate evaluation metrics.
                self.evaluation.evaluate(outputs.view(-1), data.y.view(-1))

            epoch_loss = running_loss / len(dataloader)

            # Update learning rate.
            if self.scheduler is not None and not backpropagate:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()
                self.logger.debug(
                    f"Epoch {epoch}: learning rate = {self.optimizer.param_groups[0]['lr']}"
                )

            self.writer.add_scalar(f"{dataset} loss", epoch_loss, epoch)
            return epoch_loss

    def _model_step(self, model, data, backpropagate: bool):
        self.optimizer.zero_grad()
        outputs = model(data)

        loss = self.criterion(outputs.view(-1), data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs
