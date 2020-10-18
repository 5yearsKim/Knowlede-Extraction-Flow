import torch
import os
from .trainer import Trainer
from .utils import to_one_hot, AverageMeter

class ExtractorTrainer(Trainer):
    def __init__(self, model, optimizer, train_loader, dev_loader, num_class=2, label_smoothe=0., best_save_path="ckpts/"):
        self.num_class = num_class
        criterion=None
        super(ExtractorTrainer, self).__init__(model, optimizer, criterion, train_loader, dev_loader, best_save_path)
        self.label_smoothe = label_smoothe

    def train_step(self, x, label, loss_meter):
        x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
        self.optimizer.zero_grad()
        loss = -torch.mean(self.model(x, label, smoothing=self.label_smoothe))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.flow.parameters(), 1)
        self.optimizer.step()
        loss_meter.update(loss.item())

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for x, label in self.dev_loader:
                x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
                acc = self.model.get_acc(x, label)
                acc_meter.update(acc, x.size(0))
            print(f"[{epoch} epoch Validation]: acc : {acc_meter.avg}\n")
        if acc_meter.avg > self.val_best:
            path = os.path.join(self.best_save_path, "best.pt")
            self.save(path) 



    def save(self, save_path):
        torch.save({
            'flow_state': self.model.flow.state_dict(),
            }, save_path)
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.flow.load_state_dict(save_dict['flow_state'])
        # self.model.classifier.load_state_dict(save_dict['classifier_state_dict'])


           
class AidedExtractorTrainer(ExtractorTrainer):
    def __init__(self, model, optimizer, train_loader, dev_loader, aided_loader, num_class=2, label_smoothe=0., best_save_path="ckpts"):
        super(AidedExtractorTrainer, self).__init__(model, optimizer, train_loader, dev_loader, num_class, label_smoothe, best_save_path)
        self.aided_loader = aided_loader

    def aided_train(self):
        loss_meter = AverageMeter()
        for i, (x, label) in enumerate(self.aided_loader):
            x, label = self.preprocess(x, label)
            x, label = x.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            log_ll = self.model.flow.log_prob(x, label)
            loss = -torch.mean(log_ll)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.flow.parameters(), 1.)
            self.optimizer.step()
            loss_meter.update(loss.item())
            # if i % 20 == 0 :
            #     print(f"loss = {loss_meter.avg}")
        print(f"##aided loss : {loss_meter.avg}")
    
    def on_epoch_end(self):
        self.model.flow.train()

    def on_epoch_end(self):
        self.aided_train()
         
    # def train_step(self, x, label, loss_meter ):
    #     pass
   
    def preprocess(self, x, label):
        label = to_one_hot(label, self.num_class)
        return x, label

        
