import torch
from torch import nn
import torch.nn.functional as F

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

class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            # print(param)
            param.requires_grad = False
            # print(param)
        dfs_freeze(child)


def to_one_hot(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

if __name__ == "__main__":
    label = torch.tensor([1, 2, 0, 2, 2])
    y = to_one_hot(label, 4)
    print(y)