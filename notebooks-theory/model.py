from typing import Union, List
from enum import Enum, auto

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data.dataloader import DataLoader

from evaluation import ModelEvaluation


class ModelType(Enum):
    GRAPH = auto()
    DENSE = auto()


class Model():
    def __init__(
        self,
        model,
        trainloader: DataLoader,
        valloader: DataLoader,
        writer: SummaryWriter,
        epochs: int,
        learning_rate: float,
        momentum: float,
        optimizer: torch.optim.Optimizer,
        criterion,
        weight_decay: float,
        validation_frequency: int
    ):
        # Already initiated model.
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        self.optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = criterion
        self.evaluation = ModelEvaluation(writer)

        self.epochs = epochs
        self.validation_frequency = validation_frequency

        self.momentum = momentum
        self.weight_decay = weight_decay


    def train(self):
        for epoch in range(self.epochs):
            # Train epoch.
            self._epoch_step(self.trainloader, epoch=epoch)

            # Evaluate epoch.
            if (epoch+1) % self.validation_frequency == 0:
                self.evaluation.reset()
                self._epoch_step(self.valloader, epoch=epoch, evaluate=True)
                self.evaluation.log_evaluation(epoch)


    def _epoch_step(self, dataloader, epoch, evaluate=False):
            running_loss = 0.
            if evaluate:
                self.model.eval()
            else:
                self.model.train()

            for data in dataloader:
                loss, outputs = self._model_step(data, backpropagate=evaluate)
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


    def _model_step(self, data, backpropagate=True):
        self.optimizer.zero_grad()
        outputs = self.model(data)

        loss = self.criterion(outputs, data.y)
        if backpropagate:
            loss.backward()
            self.optimizer.step()

        return loss.item(), outputs
