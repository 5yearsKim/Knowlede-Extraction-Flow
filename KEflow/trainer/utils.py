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

def label_smoothe(label, smoothing=0.):
    assert smoothing > 0 and smoothing < 0.5
    bs, num_classes = label.size()

    label = label * (1 - smoothing) + smoothing / (num_classes - 1) * (1. - label)
    return label

def img_reg_loss(x):
    x = x * 0.95
    loss = - x.log() - (1 - x).log() + 2 * torch.tensor(0.5).log()
    loss = loss * 100 
    return loss.reshape(x.size(0), -1).mean(1)


def to_one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

'''
adapted from https://github.com/NVlabs/DeepInversion/blob/master/deepinversion.py
'''
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


if __name__ == "__main__":
    label = torch.tensor([1, 2, 0, 2, 2])
    y = to_one_hot(label, 4)
    y = label_smoothe(y, smoothing=0.1)
    print(y)