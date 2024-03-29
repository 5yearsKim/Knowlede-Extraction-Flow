import torch
from torch import nn
from .utils import AverageMeter, to_one_hot

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, dev_loader):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.dev_loader = dev_loader

    def train(self, epochs, print_freq=10, val_freq=1):
        self.on_train_start()
        loss_meter = AverageMeter()
        for epoch in range(epochs):
            self.model.train()
            self.on_epoch_start()
            loss_meter.reset()
            for i, (x, label) in enumerate(self.train_loader):
                self.train_step(x, label, loss_meter)
                if i%print_freq == 0:
                    print(f'iter {i} : loss = {loss_meter.avg}')
            print(f"*epoch {epoch}: loss = {loss_meter.avg}")
            if i%val_freq ==0:
                with torch.no_grad():
                    self.validate()

    def train_step(self, x, label, loss_meter):
        x, label = x.to(self.device), label.to(self.device)
        self.optimizer.zero_grad()
        logit = self.model(x)
        loss = self.criterion(logit, label)
        loss.backward()
        self.optimizer.step()
        loss_meter.update(loss.item())

    def validate(self):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for x, label in self.dev_loader:
            x, label = x.to(self.device), label.to(self.device)
            logit = self.model(x)
            # loss calculation
            loss = self.criterion(logit, label)
            loss_meter.update(loss.to('cpu').item())
            # Accuracy calculation
            _, predicted = torch.max(logit.to('cpu'), 1)
            hit_rate = float(predicted.eq(label).sum()) / float(len(label))
            acc_meter.update(hit_rate, n=len(label))
        print(f"[Validation]: loss : {loss_meter.avg}, acc : {acc_meter.avg}\n")

    def on_train_start(self):
        self.model.train()

    def on_epoch_start(self):
        self.model.train()
            
    def save(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }, save_path)
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.load_state_dict(save_dict['model_state_dict'])

        