import torch

class Trainer:
    def __init__(self, model, train_loader, dev_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        
    def train(self, epoch):
        for x, label in self.train_loader:
            