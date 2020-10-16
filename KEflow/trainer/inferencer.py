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
        _label = to_one_hot(label, self.num_class)
        x, _ = self.model(z, _label, reverse=True)
        return x, label

    def save_pic(self, x, path):
        torchvision.utils.save_image(x, path)

    def amplify(self, x):
        amp = 1.4
        x = torch.clamp(x * amp - (amp-1) , 0, 1)
        return x
