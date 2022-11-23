import torch
import torch.nn as nn

"""Affine coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, in_out_dim, cond_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.
        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config
        self.scale = nn.Parameter(torch.ones(in_out_dim//2))

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.cond_block = nn.Sequential(
            nn.Linear(cond_dim, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, (in_out_dim//2) * 2)
    

    def forward(self, x, cond, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off) + self.cond_block(cond)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        off_ = self.out_block(off_)
        s, shift = off_.split(W//2, dim=1)
        
        log_scale = self.scale * torch.tanh(s)

        if reverse:
            on = (on - shift) * torch.exp(-log_scale)
        else:
            on = torch.exp(log_scale) * on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        
        return x.reshape((B, W)), torch.sum(log_scale, dim=1)

"""NICE main model.
"""
class AffineNICE(nn.Module):
    def __init__(self, prior, coupling, 
        in_out_dim, cond_dim, mid_dim, hidden, mask_config=0):
        super(AffineNICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim, 
                     cond_dim= cond_dim, 
                     mid_dim=mid_dim, 
                     hidden=hidden, 
                     mask_config=(mask_config+i)%2) \
            for i in range(coupling)])

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.uniform_(-0.04, 0.04)
        #         m.bias.data.fill_(0.)

    def g(self, z, cond):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        log_det_J = 0
        for i in reversed(range(len(self.coupling))):
            z, log_scale = self.coupling[i](z, cond, reverse=True)
            log_det_J += -log_scale
        return z, log_det_J

    def f(self, x, cond):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        log_det_J = 0
        for i in range(len(self.coupling)):
            x, log_scale = self.coupling[i](x, cond)
            log_det_J += log_scale
        return x, log_det_J

    def log_prob(self, x, cond):
        """Computes data log-likelihood.
        (See Section 3.3 in the NICE paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x, cond)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, cond):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        size = cond.size(0)
        z = self.prior.sample((size, self.in_out_dim))
        return self.g(z, cond)

    def forward(self, x, cond, reverse=False):
        if reverse:
            return self.g(x, cond)
        else:
            return self.f(x, cond)




if __name__ == "__main__":
    prior = torch.distributions.Normal(
        torch.tensor(0.), torch.tensor(1.))
    coupling = 6
    in_out_dim = 2
    cond_dim = 2
    mid_dim = 20
    hidden = 2

    x = torch.tensor([[3., 2.]])
    cond = torch.tensor([[1., 0.]])

    model = NICE(prior, coupling, in_out_dim, cond_dim, mid_dim, hidden)

    print(model(x, cond))

