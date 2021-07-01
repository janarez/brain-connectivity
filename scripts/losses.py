class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        num = torch.sum(2*predict*target, dim=1) + 1
        den = torch.sum(predict + target, dim=1) + 1
        return torch.sum(1 - num / den)
