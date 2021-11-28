"""
Module contains class for calculating model metrics and logging them to tensorboard,
plus related util functions.
"""
import os
from collections import defaultdict

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from .general_utils import get_logger


class ModelEvaluation:
    """
    Class for calculating model metrics and logging them to tensorboard.
    """

    def __init__(self, log_folder: str):
        self.logger = get_logger(
            "evaluation", os.path.join(log_folder, "evaluation.txt")
        )
        self.train_results, self.val_results = [], []
        self.dev_results, self.test_results = [], []
        self._epoch_reset()

    def set_fold(self, _):
        self.train_results.append(defaultdict(list))
        self.val_results.append(defaultdict(list))
        self.dev_results.append(defaultdict(list))
        self.test_results.append(defaultdict(list))

    def evaluate(self, predicted, labels):
        """
        Calculates confusion matrix between predicted and true labels for single batch.
        Saves running totals.
        """
        pred_positives = predicted == 1
        label_positives = labels == 1

        self.tp += (pred_positives & label_positives).sum().item()
        self.tn += (~pred_positives & ~label_positives).sum().item()
        self.fp += (pred_positives & ~label_positives).sum().item()
        self.fn += (~pred_positives & label_positives).sum().item()

        self.total += len(labels)

    def log_evaluation(self, epoch, dataset: str, writer: SummaryWriter):
        """
        Calculates metrics on aggregated confusion matrices.
        Saves and logs them to tensorboard.
        """
        # Calculate.
        accuracy = (self.tp + self.tn) / self.total
        recall = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        precision = self.tp / (self.tp + self.fn)

        # Log.
        writer.add_scalar(f"{dataset} accuracy", accuracy, epoch)
        writer.add_scalar(f"{dataset} precision", recall, epoch)
        writer.add_scalar(f"{dataset} recall", precision, epoch)
        self.logger.debug(f"Epoch {epoch}: {dataset} accuracy = {accuracy}")
        self.logger.debug(f"Epoch {epoch}: {dataset} precision = {precision}")
        self.logger.debug(f"Epoch {epoch}: {dataset} recall = {recall}")

        # Save.
        results = self._get_results(dataset)
        results["accuracy"].append(accuracy)
        results["epoch"].append(epoch)
        results["recall"].append(recall)
        results["precision"].append(precision)

        # Reset.
        self._epoch_reset()

    def _get_results(self, dataset: str, index=-1):
        if dataset == "train":
            return self.train_results[index]
        elif dataset == "val":
            return self.val_results[index]
        elif dataset == "test":
            return self.test_results[index]
        elif dataset == "dev":
            return self.dev_results[index]
        else:
            raise ValueError(
                f"Incorrect `dataset` value, got {dataset}, accept: train, test, dev, val."
            )

    def _epoch_reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        # How many times was evaluation run?
        self.total = 0

    def _aggregate_results(self, results):
        "Averages over list of dictionaries with results."
        agg_results = {}
        for k in results[0].keys():
            val_list = [res[k] for res in results]
            # Take mean and standard deviation across runs.
            agg_results[k] = (
                np.mean(val_list, axis=0),
                np.std(val_list, axis=0),
            )

        return agg_results

    def get_experiment_results(self, dataset: str):
        results = self._get_results(dataset, index=slice(None))
        return self._aggregate_results(results)
