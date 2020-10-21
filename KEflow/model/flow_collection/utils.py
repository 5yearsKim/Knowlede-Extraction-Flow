import torch

def dequantize_to_logit(x, bound=0.95):
    y = (x * 255. + 5*torch.rand_like(x)) / 256.
    y = y.clamp(0, 1)
    y = (2 * y - 1) * bound
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()
    return y