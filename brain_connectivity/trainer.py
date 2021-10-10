from typing import Union, List

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data.dataloader import DataLoader

from .evaluation import ModelEvaluation


class Trainer():
    def __init__(
        self,
        model,
        trainloader: DataLoader,
        valloader: DataLoader,
        writer: SummaryWriter,
        epochs: int,
        validation_frequency: int,
        fc_matrix_plot_frequency: int,
        fc_matrix_plot_sublayer: int,
        optimizer: torch.optim.Optimizer,
        optimizer_kwargs: dict,
        step_size: int,
        gamma: float,
        criterion,
    ):
        # Already initiated model.
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.criterion = criterion
        self.writer = writer
        self.evaluation = ModelEvaluation(writer)

        self.epochs = epochs
        self.validation_frequency = validation_frequency

        self.fc_matrix_plot_frequency = fc_matrix_plot_frequency
        self.fc_matrix_plot_sublayer = fc_matrix_plot_sublayer


    def train(self):
        for epoch in range(self.epochs):
            # Train epoch.
            self._epoch_step(self.trainloader, epoch=epoch)

            # Evaluate epoch.
            if (epoch+1) % self.validation_frequency == 0:
                self.evaluation.reset()
                self._epoch_step(self.valloader, epoch=epoch, evaluate=True)
                self.evaluation.log_evaluation(epoch)

            # Plot connectivity matrix.
            if (epoch+1) % self.fc_matrix_plot_frequency == 0:
                self.model.plot_fc_matrix(epoch, sublayer=self.fc_matrix_plot_sublayer)


    def _epoch_step(self, dataloader: DataLoader, epoch: int, evaluate: bool = False):
            running_loss = 0.
            if evaluate:
                self.model.eval()
            else:
                self.model.train()

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
            self.writer.add_scalar(
                f'{"Validation" if evaluate else "Training"} loss',
                epoch_loss, epoch
            )


    def _model_step(self, data, backpropagate: bool = True):
        self.optimizer.zero_grad()
        outputs = self.model(data)

        loss = self.criterion(outputs, data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs
