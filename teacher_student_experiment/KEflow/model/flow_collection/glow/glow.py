import torch
import torch.nn as nn
import torch.nn.functional as F

from .act_norm import ActNorm
from .coupling import Coupling
from .inv_conv import InvConv

from ..utils import dequantize_to_logit

class Glow(nn.Module):
    """Glow Model
    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).
    Args:
        mid_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, mid_channels, cond_channels, num_levels, num_steps):
        super(Glow, self).__init__()

        self.prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
        # Use bounds to rescale images before converting to logits, not learned
        self.flows = _Glow(in_channels=4 * in_channels,  # in channels after squeeze
                           mid_channels=mid_channels,
                           cond_channels=cond_channels,
                           num_levels=num_levels,
                           num_steps=num_steps)

    def forward(self, x, cond, reverse=False):
        sldj = torch.zeros(x.size(0), device=x.device)
        if not reverse:
            x, ldj = dequantize_to_logit(x)
            sldj += ldj
        x = squeeze(x)
        x, ldj = self.flows(x, cond, sldj, reverse)
        sldj += ldj
        x = squeeze(x, reverse=True)
        if reverse:
            x, ldj = dequantize_to_logit(x, reverse=True)
            sldj += ldj
        return x, sldj

    def log_prob(self, x, cond):
        z, sldj = self.forward(x, cond)
        log_ll = self.prior.log_prob(z).sum(-1).sum(-1).sum(-1)
        return log_ll + sldj





class _Glow(nn.Module):
    """Recursive constructor for a Glow model. Each call creates a single level.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in hidden layers of each step.
        num_levels (int): Number of levels to construct. Counter for recursion.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, in_channels, mid_channels, cond_channels, num_levels, num_steps):
        super(_Glow, self).__init__()
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels,
                                              mid_channels=mid_channels,
                                              cond_channels=cond_channels)
                                    for _ in range(num_steps)])

        if num_levels > 1:
            self.next = _Glow(in_channels=2 * in_channels,
                              mid_channels=mid_channels,
                              cond_channels=cond_channels,
                              num_levels=num_levels - 1,
                              num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, cond, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, cond, sldj, reverse)

        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, cond, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)

        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, cond, sldj, reverse)

        return x, sldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels, cond_channels):
        super(_FlowStep, self).__init__()

        # Activation normalization, invertible 1x1 convolution, affine coupling
        # self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels, cond_channels)

    def forward(self, x, cond, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, cond, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            # x, sldj = self.norm(x, sldj, reverse)
        else:
            # x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, cond, sldj, reverse)

        return x, sldj


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.
    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.
    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x

