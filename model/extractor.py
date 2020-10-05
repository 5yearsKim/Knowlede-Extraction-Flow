import torch
from torch import nn

class Extractor(nn.Module):
    def __init__(self, flow, classifier):
        super(Extractor, self).__init__()
        self.flow = flow
        self.classifier = classifier
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, cond):
        bs, cond_dim = cond.size(0), cond.size(1)
        y, log_det_J = self.flow(x, cond, reverse=True) 
        confidence = self.LogSoftmax(self.classifier(y))
        log_ll = torch.bmm(confidence.view(bs, 1, cond_dim), cond.view(bs, cond_dim, 1)).view(bs)
        reg = torch.sum(torch.square(y), dim=1)
        # print(log_det_J, log_ll)
        return 1* log_det_J + 60*log_ll - 0.001 * reg


