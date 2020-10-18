import torch
from torch import nn
from KEflow.model.utils import label_smoothe, img_reg_loss



class Extractor(nn.Module):
    def __init__(self, flow, classifier, alpha=0.04, beta=2.):
        super(Extractor, self).__init__()
        self.flow = flow
        self.classifier = classifier
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x, cond, smoothing=0.):
        y, log_det_J = self.flow(x, cond, reverse=True) 
        reg = img_reg_loss(y) #regularization loss
        y = self.normalize_images(y)
        confidence = self.LogSoftmax(self.classifier(y))
        if smoothing != 0.:
            cond = label_smoothe(cond, smoothing)
        bs, cond_dim = cond.size(0), cond.size(1)
        log_ll = torch.bmm(confidence.view(bs, 1, cond_dim), cond.view(bs, cond_dim, 1)).view(bs)
        # print(log_det_J, log_ll)
        return log_ll  + self.alpha * log_det_J - self.beta * reg

    def get_acc(self, x, cond):
        y, log_det_J = self.flow(x, cond, reverse=True) 
        y = self.normalize_images(y)
        _, predicted = torch.max(self.classifier(y), 1)
        _, cond = torch.max(cond, 1)
        hit_rate = float(predicted.eq(cond).sum())/ float(cond.size(0))
        return hit_rate

    @staticmethod
    def normalize_images(layer, mean=None, std=None):
        """
        Normalize images into zero-mean and unit-variance.
        """
        if mean is None:
            mean = layer.mean(dim=(2, 3), keepdim=True)
        if std is None:
            std = layer.view((layer.size(0), layer.size(1), -1)) \
                .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std
