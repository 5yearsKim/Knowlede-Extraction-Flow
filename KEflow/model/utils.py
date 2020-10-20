import torch
from torch import nn
from .classifier import BasicCNN, LeNet5, ResNet, vgg11_bn

def prepare_classifier(cls_type, nc, n_class):
    if cls_type == "BASICCNN":
        return BasicCNN(nc, n_class)
    elif cls_type == "LENET5":
        return LeNet5(nc, n_class)
    elif cls_type == "RESNET":
        return ResNet(nc, n_class)
    elif cls_type == "VGG":
        return vgg11_bn(pretrained=True)
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
    x = x * 0.98
    loss = - x.log() - (1 - x).log() + 2 * torch.tensor(0.5).log()
    loss = (loss * 3) ** 2
    return loss.view(x.size(0), -1).mean(1)

def weights_init(m, gain=0.5):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    