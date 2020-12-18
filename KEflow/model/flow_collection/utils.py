import torch
import torch.nn.functional as F

def dequantize_to_logit(x, reverse=False, bound=0.9):
    if reverse:
        y = torch.sigmoid(x)
        ldj = torch.log(y * (1 - y))
        sldj = ldj.flatten(1).sum(-1)
        return y, sldj
    else:
        y = (x * 255. + 5*torch.rand_like(x)) / 256.
        y = y.clamp(0, 1)
        y = (2 * y - 1) * bound
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) 
            # - F.softplus((1. - bound).log() - bound.log())
        sldj = ldj.flatten(1).sum(-1)    
        
        return y, sldj