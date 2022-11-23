import torch
from torch import nn
import os
from .utils import AverageMeter, to_one_hot
import wandb

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, dev_loader, best_save_path="ckpts/", use_wandb=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.val_best = 0.
        self.best_save_path = best_save_path
        self.use_wandb = use_wandb

    def train(self, epochs, print_freq=10, val_freq=1, use_wandb=False):
        loss_meter = AverageMeter()
        for epoch in range(epochs):
            self.model.train()
            self.fix_bn_layer()
            loss_meter.reset()
            for i, (x, label) in enumerate(self.train_loader):
                self.train_step(x, label, loss_meter)
                if i != 0 and i%print_freq == 0:
                    print(f'iter {i} : loss = {loss_meter.avg}')
            print(f"*epoch {epoch}: loss = {loss_meter.avg}")
            if i%val_freq ==0:
                with torch.no_grad():
                    val_best = self.validate(epoch)

    def train_step(self, x, label, loss_meter):
        x, label = x.to(self.device), label.to(self.device)
        self.optimizer.zero_grad()
        logit = self.model(x)
        loss = self.criterion(logit, label)
        loss.backward()
        self.optimizer.step()
        loss_meter.update(loss.item())

    def validate(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for x, label in self.dev_loader:
            x, _label = x.to(self.device), to_one_hot(label, 10).to(self.device)
            logit = self.model(x)
            prob = torch.nn.functional.softmax(logit, dim=-1)
            # loss calculation
            loss = self.criterion(prob, _label)
            loss_meter.update(loss.to('cpu').item())
            # Accuracy calculation
            _, predicted = torch.max(logit.to('cpu'), 1)
            label = label.to('cpu')
            hit_rate = float(predicted.eq(label).sum()) / float(len(label))
            acc_meter.update(hit_rate, n=len(label))
        print(f"[{epoch} epoch Validation]: loss : {loss_meter.avg}")
        if acc_meter.avg > self.val_best:
            self.val_best = acc_meter.avg
            path = os.path.join(self.best_save_path, "best.pt")
            self.save(path)
            print(f"acc : {acc_meter.avg}  BEST\n")
        else:
            print(f"acc : {acc_meter.avg}  \n")
        if self.use_wandb:
            wandb.log({'epoch':epoch, 'acc':acc_meter.avg})

    def fix_bn_layer(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    def save(self, save_path):
        torch.save({
            'model_state': self.model.state_dict(),
            }, save_path)
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.load_state_dict(save_dict['model_state'])

        
