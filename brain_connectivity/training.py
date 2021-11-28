"""
Contains general training class for training any model.
"""
import os
from typing import Optional, Tuple

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data.dataloader import DataLoader
from torchinfo.torchinfo import summary

from .evaluation import ModelEvaluation
from .general_utils import close_logger, get_logger
from .model import Model


class Trainer:
    # TODO: Write class docstring.
    def __init__(
        self,
        log_folder,
        epochs: int,
        validation_frequency: int,
        fc_matrix_plot_frequency: int,
        fc_matrix_plot_sublayer: int,
        criterion,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_kwargs: Optional[dict] = None,
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

        self.fc_matrix_plot_frequency = fc_matrix_plot_frequency
        self.fc_matrix_plot_sublayer = fc_matrix_plot_sublayer

        # Log training setting.
        self.logger.info(f"Optimizer: {optimizer}, {optimizer_kwargs}")
        self.logger.info(f"Scheduler: {scheduler} {scheduler_kwargs}")
        self.logger.info(f"Criterion: {self.criterion}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Validation frequency: {self.validation_frequency}")
        self.logger.info(
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
        self.writer = SummaryWriter(os.path.join(self.log_folder), fold)

        for epoch in range(self.epochs):
            # Train epoch.
            evaluate = (epoch + 1) % self.validation_frequency == 0
            self._epoch_step(
                model,
                trainloader,
                epoch=epoch,
                dataset=train_dataset,
                evaluate=evaluate,
            )
            if evaluate:
                self.evaluation.log_evaluation(
                    epoch, train_dataset, self.writer
                )

            # Evaluate epoch.
            if (epoch + 1) % self.validation_frequency == 0:
                self._epoch_step(
                    model,
                    evalloader,
                    epoch=epoch,
                    dataset=eval_dataset,
                    evaluate=True,
                )
                self.evaluation.log_evaluation(epoch, eval_dataset, self.writer)

            # Plot connectivity matrix.
            if (epoch + 1) % self.fc_matrix_plot_frequency == 0:
                model.plot_fc_matrix(
                    epoch, sublayer=self.fc_matrix_plot_sublayer
                )

    def get_results(self, train_dataset, eval_dataset):
        train_res = self.evaluation.get_experiment_results(train_dataset)
        eval_res = self.evaluation.get_experiment_results(eval_dataset)

        # Close all loggers used in single inner CV run.
        close_logger("dataset")
        close_logger("evaluation")
        close_logger("trainer")
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
                predicted = outputs.argmax(dim=1)
                # labels = torch.nonzero(data.y, as_tuple=True)[1]
                labels = data.y.view(-1)
                self.evaluation.evaluate(predicted, labels)

        # Update learning rate.
        if self.scheduler is not None and backpropagate:
            self.scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        self.writer.add_scalar(f"{dataset} loss", epoch_loss, epoch)

    def _model_step(self, model, data, backpropagate: bool = True):
        self.optimizer.zero_grad()
        outputs = model(data)

        loss = self.criterion(outputs, data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs
