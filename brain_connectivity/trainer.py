"""
Contains general training class for training any model.
"""
import os

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data.dataloader import DataLoader
from torchinfo.torchinfo import summary

from .evaluation import ModelEvaluation


class Trainer:
    def __init__(
        self,
        log_folder,
        # Already initiated model.
        model,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs: int,
        validation_frequency: int,
        fc_matrix_plot_frequency: int,
        fc_matrix_plot_sublayer: int,
        criterion,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_kwargs: dict = {},
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.scheduler = scheduler(self.optimizer, **scheduler_kwargs) if scheduler is not None else None
        self.criterion = criterion

        self.writer = SummaryWriter(log_folder)
        self.evaluation = ModelEvaluation(self.writer)
        self.log_folder = log_folder

        self.epochs = epochs
        self.validation_frequency = validation_frequency

        self.fc_matrix_plot_frequency = fc_matrix_plot_frequency
        self.fc_matrix_plot_sublayer = fc_matrix_plot_sublayer

        # Save to file before training.
        self._log()

    def _log(self):
        """
        Logs all important training parameters and model to a file.
        """
        with open(os.path.join(self.log_folder, "trainer.txt"), "w", encoding="utf-8") as f:
            f.write(self.__dict__.__str__() + "\n")
            f.write(self.model.__str__() + "\n\n")
            f.write(str(summary(self.model)))

    def train(self):
        "Runs training, all relevant arguments must be provided on class creation."
        for epoch in range(self.epochs):
            # Train epoch.
            evaluate = (epoch + 1) % self.validation_frequency == 0
            dataset = "train"
            self._epoch_step(self.trainloader, epoch=epoch, dataset=dataset, evaluate=evaluate)
            if evaluate:
                self.evaluation.log_evaluation(epoch, dataset=dataset)

            # Evaluate epoch.
            if (epoch + 1) % self.validation_frequency == 0:
                dataset = "val"
                self._epoch_step(self.valloader, epoch=epoch, dataset=dataset, evaluate=True)
                self.evaluation.log_evaluation(epoch, dataset=dataset)

            # Plot connectivity matrix.
            if (epoch + 1) % self.fc_matrix_plot_frequency == 0:
                self.model.plot_fc_matrix(epoch, sublayer=self.fc_matrix_plot_sublayer)

            # Update learning rate.
            if self.scheduler is not None:
                self.scheduler.step()

    def _epoch_step(self, dataloader: DataLoader, epoch: int, dataset: str, evaluate: bool):
        running_loss = 0.0
        if dataset == "val":
            self.model.eval()
        elif dataset == "train":
            self.model.train()
        else:
            raise ValueError("Dataset must be either 'val' or 'train'")

        for data in dataloader:
            loss, outputs = self._model_step(data, backpropagate=not evaluate)
            running_loss += loss

            # Calculate evaluation metrics.
            if evaluate:
                predicted = outputs.argmax(dim=1)
                # labels = torch.nonzero(data.y, as_tuple=True)[1]
                labels = data.y.view(-1)
                self.evaluation.evaluate(predicted, labels)

        epoch_loss = running_loss / len(dataloader)
        self.writer.add_scalar(f"{dataset} loss", epoch_loss, epoch)

    def _model_step(self, data, backpropagate: bool = True):
        self.optimizer.zero_grad()
        outputs = self.model(data)

        loss = self.criterion(outputs, data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs
