import torch
from torch import nn
from utils import AverageMeter

class Trainer:
    def __init__(self, model, train_loader, dev_loader, print_freq=100, val_freq=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.criterion = nn.CrossEntropyLoss()
        self.print_freq = print_freq
        self.val_freq = val_freq

    def train(self, epochs):
        loss_meter = AverageMeter()
        for i in range(epochs):
            for x, label in self.train_loader:
                self.train_step(x, label, loss_meter)
            if i%val_freq ==0:
                with torch.no_grad():
                    self.validate

    def train_step(self, x, label, loss_meter):
        self.optimizer.zero_grad()
        logit = self.model(x)
        loss = self.criterion(logit, label)
        loss_meter.update(loss.item())
        loss.backward()
        self.optimizer.step()

    def validate(self):
        loss_meter = AverageMeter()
        for x, label in self.dev_loader:
            logit = self.model(x)
            loss = self.criterion(logit, label)
            loss_meter.update(loss.item())
        



            
