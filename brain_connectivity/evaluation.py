"""
Module contains class for calculating model metrics and logging them to tensorboard,
plus related util functions.
"""
import os
from collections import defaultdict

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class ModelEvaluation:
    """
    Class for calculating model metrics and logging them to tensorboard.
    """

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.train_results = defaultdict(list)
        self.val_results = defaultdict(list)
        self._reset()

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

    def log_evaluation(self, epoch, dataset: str):
        """
        Calculates metrics on aggregated confusion matrices.
        Saves and logs them to tensorboard.
        """
        # Calculate.
        accuracy = (self.tp + self.tn) / self.total
        recall = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        precision = self.tp / (self.tp + self.fn)

        # Log.
        self.writer.add_scalar(f"{dataset} accuracy", accuracy, epoch)
        self.writer.add_scalar(f"{dataset} precision", recall, epoch)
        self.writer.add_scalar(f"{dataset} recall", precision, epoch)

        # Save.
        results = self.train_results if dataset == "train" else self.val_results
        results["accuracy"].append(accuracy)
        results["epoch"].append(epoch)
        results["recall"].append(recall)
        results["precision"].append(precision)

        # Reset.
        self._reset()

    def _reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        # How many times was evaluation run?
        self.total = 0


def aggregate_results(results):
    """
    Averages over list of dictionaries with results.
    """
    agg_results = {}
    for k in results[0].keys():
        val_list = [res[k] for res in results]
        # Take mean and standard deviation across runs.
        agg_results[k] = [*zip(np.mean(val_list, axis=0), np.std(val_list, axis=0))]

    return agg_results


def log_results(results, log_folder, log_filename):
    """
    Writes dictionary with results to a file.
    """
    with open(os.path.join(log_folder, f"{log_filename}.txt"), "w", encoding="utf-8") as f:
        for key, value_list in results.items():
            f.write(f"{key}\n")
            for value in value_list:
                f.write(f"{value}\n")
