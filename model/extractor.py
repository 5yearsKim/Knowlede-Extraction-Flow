import torch
from torch import nn

class Extractor(nn.Module):
    def __init__(self, flow, classifier):
        super(Extractor, self).__init__()
        self.flow = flow
        self.classifier = classifier
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(x, cond):
        y, log_det_J = self.flow(x, cond) 
        confidence = self.LogSoftmax(self.classifier(y))
        log_ll = torch.einsum(confidence, cond)
        return log_det_J + log_ll


