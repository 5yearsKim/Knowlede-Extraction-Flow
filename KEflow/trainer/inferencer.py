import torch
import torchvision
from .utils import to_one_hot

class Inferencer:
    def __init__(self, model, size, num_class=10):
        self.model = model
        self.num_class=num_class
        self.size = size

    def inference(self, temp=0.05):
        label = torch.arange(0, 10)
        _label = to_one_hot(label, self.num_class)
        z = torch.randn(10 ,*self.size) * temp
        x, _ = self.model(z, _label, reverse=True)
        return x, label

    def save_pic(self, x, path):
        torchvision.utils.save_image(x, path)

    def amplify(self, x):
        amp = 1.4
        x = torch.clamp(x * amp - (amp-1) , 0, 1)
        return x
