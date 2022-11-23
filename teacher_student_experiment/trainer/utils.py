import torch
from torch import nn

class AverageMeter:
    """code from TNT"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logit, label):
        return self.loss(self.log_softmax(logit), label)
