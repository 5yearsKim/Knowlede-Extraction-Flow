import torch
import torch.nn as nn
import torch.nn.functional as F

from .act_norm import ActNorm


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
        cond_channels (int): Number of dimension of condition
    """
    def __init__(self, in_channels, mid_channels, cond_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, mid_channels, 2 * in_channels, cond_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1) * 1.05)

    def forward(self, x, cond, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, cond)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.clamp(torch.tanh(s), -0.95, 0.95)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, out_channels, cond_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_conv = nn.Conv2d(in_channels + cond_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        mid_layers = []
        for i in range(2):
            mid_layers += [norm_fn(mid_channels),
                           nn.ReLU(),
                           nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=True),
                          ]   
        self.mid_layers = nn.Sequential( *mid_layers )

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, cond):
        cond = cond.unsqueeze(-1).unsqueeze(-1)
        cond = cond.repeat(1, 1, x.size(-2), x.size(-1))

        x = torch.cat((x, cond), dim=1)
        x = self.in_conv(x)
        x = self.mid_layers(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
