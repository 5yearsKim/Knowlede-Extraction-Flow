import torch
from utils import to_one_hot

class Inferencer:
    def __init__(self, model, dataloader, cond_dim=2):
        self.model = model.eval()
        self.dataloader = dataloader
        self.cond_dim = 2

    def inference(self):
        out_bin = []
        label_bin = []
        for z, label in self.dataloader:
            cond = to_one_hot(label, self.cond_dim)        
            out, _ = self.model(z, cond, reverse=True)
            out_bin.append(out)
            label_bin.append(label)
        # print(out_bin)
        return torch.cat(out_bin, dim=0) , torch.cat(label_bin, dim=0)
