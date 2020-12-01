from torch import nn
from .classifier import BasicCNN, LeNet5, ResNet, vgg11_bn, WideResidualNetwork
from .flow_collection import AffineNICE, Glow, RealNVP

def prepare_classifier(cls_type, nc, n_class):
    if cls_type == "BASICCNN":
        return BasicCNN(nc, n_class)
    elif cls_type == "LENET5":
        return LeNet5(nc, n_class)
    elif cls_type == "RESNET":
        return ResNet(nc, n_class)
    elif cls_type == "VGG":
        return vgg11_bn(pretrained=True)
    elif cls_type == "WRN":
        return WideResidualNetwork((1, nc, 32, 32), n_class, base_channels=16, widening_factor=10, drop_rate=0.1, depth=28)
    else:
        raise ValueError(f"{cls_type} not supported!")

def prepare_flow(flow_type, nc, n_class, im_size=32):
    if flow_type == "NICE":
        return AffineNICE(nc, im_size, 4, n_class, 1500, 2)
    elif flow_type == "GLOW":
        return Glow(nc, 32, n_class, 3, 8)
    elif flow_type == "REALNVP":
        return RealNVP(2, nc, 32, num_blocks=5)
    else:
        raise ValueError(f"{flow_type} not supported!")


def weights_init(m, gain=0.5):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=gain)

