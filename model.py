import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, dim_in=2, dim_out=2,  n_hidden=2, dim_hidden=32):
        super(Classifier, self).__init__()
        self.n_hidden = n_hidden
        self.in_layer = nn.Linear(dim_in, dim_hidden)
        self.hidden = []
        for i in range(n_hidden):
            self.hidden += [nn.Linear(dim_hidden, dim_hidden)]
        self.out_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for i in range(self.n_hidden):
            x = F.relu(self.hidden[i](x))
        out = self.out_layer(x)
        return out

if __name__ == "__main__":
    from dataloader import ToyDataset
    from torch.utils.data import DataLoader
    
    toyset = ToyDataset(500)
    loader = DataLoader(toyset, batch_size=4)
    net = Classifier()
    x, y = next(iter(loader))
    print(net(x))
    

