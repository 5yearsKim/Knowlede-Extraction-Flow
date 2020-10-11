import torch
import torchvision
from .utils import to_one_hot

class Inferencer:
    def __init__(self, model, dataloader, num_class=10):
        self.model = model
        self.dataloader = dataloader
        self.num_class=num_class

    def inference(self):
        z, label = next(iter(self.dataloader))
        z = z.view(z.size(0), -1)
        z, _label = z, to_one_hot(label, self.num_class)
        x, _ = self.model(z, _label)
        return torch.sigmoid(x.view(-1, 1, 32, 32)), label

    def save_pic(self, x, path):
        torchvision.utils.save_image(x, path)
