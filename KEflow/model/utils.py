import torch
from .classifier import BasicCNN, LeNet5, ResNet

def prepare_classifier(cls_type, nc, n_class):
    if cls_type == "BASICCNN":
        return BasicCNN(nc, n_class)
    elif cls_type == "LENET5":
        return LeNet5(nc, n_class)
    elif cls_type == "RESNET":
        return ResNet(nc, n_class)
    else:
        raise ValueError(f"{cls_type} not supported!")


def dequantize_to_logit(x, bound=0.95):
    y = (x * 255. + 5*torch.rand_like(x)) / 256.
    y = y.clamp(0, 1)
    y = (2 * y - 1) * bound
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    return y

def label_smoothe(label, smoothing=0.):
    assert smoothing > 0 and smoothing < 0.5
    bs, num_classes = label.size()

    label = label * (1 - smoothing) + smoothing / (num_classes - 1) * (1. - label)
    return label

def img_reg_loss(x):
    loss = - x.log() - (1 - x).log() + 2 * torch.tensor(0.5).log()
    loss = (loss * 3) ** 2
    return loss.view(x.size(0), -1).mean(1)

