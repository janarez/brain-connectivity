from torch.utils.tensorboard.writer import SummaryWriter


class ModelEvaluation():

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.best_results = {
            'accuracy': -1
        }

        self.reset()


    def evaluate(self, predicted, labels):
        pred_positives = predicted == 1
        label_positives = labels == 1

        self.tp += (pred_positives & label_positives).sum().item()
        self.tn += (~pred_positives & ~label_positives).sum().item()
        self.fp += (pred_positives & ~label_positives).sum().item()
        self.fn += (~pred_positives & label_positives).sum().item()

        self.total += len(labels)


    def log_evaluation(self, epoch):
        # Calculate.
        accuracy = (self.tp + self.tn) / self.total
        recall = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        precision = self.tp / (self.tp + self.fn)

        # Log.
        self.writer.add_scalar('validation accuracy', accuracy, epoch)
        self.writer.add_scalar('validation precision', recall, epoch)
        self.writer.add_scalar('validation recall', precision, epoch)

        # Save if best so far.
        if self.best_results['accuracy'] < accuracy:
            self.best_results['accuracy'] = accuracy
            self.best_results['epoch'] = epoch
            self.best_results['recall'] = recall
            self.best_results['precision'] = precision


    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        # How many times was evaluation run?
        self.total = 0


    def get_best_results(self):
        return self.best_results
