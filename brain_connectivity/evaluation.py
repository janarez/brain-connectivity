from torch.utils.tensorboard.writer import SummaryWriter


class ModelEvaluation():

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
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
        self.writer.add_scalar('validation accuracy', (self.tp + self.tn) / self.total, epoch)
        self.writer.add_scalar('validation precision', self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0, epoch)
        self.writer.add_scalar('validation recall', self.tp / (self.tp + self.fn), epoch)


    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        # How many times was evaluation run?
        self.total = 0
