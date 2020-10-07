import torch
from .trainer import Trainer
from .utils import to_one_hot, AverageMeter

class ExtractorTrainer(Trainer):
    def __init__(self, model, optimizer, train_loader, dev_loader, num_class=2, label_smoothe=0.):
        self.num_class = num_class
        criterion=None
        super(ExtractorTrainer, self).__init__(model, optimizer, criterion, train_loader, dev_loader)
        self.label_smoothe = label_smoothe

    def train_step(self, x, label, loss_meter):
        x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
        self.optimizer.zero_grad()
        loss = -torch.mean(self.model(x, label, smoothing=self.label_smoothe))
        loss.backward()
        self.optimizer.step()
        loss_meter.update(loss.item())

    def validate(self):
        self.model.eval()
        loss_meter = AverageMeter()
        with torch.no_grad():
            for x, label in self.dev_loader:
                x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
                loss = - torch.mean(self.model(x, label))
                loss_meter.update(loss.to('cpu').item())
            print(f"[Validation]: loss : {loss_meter.avg}\n")

    def save(self, save_path):
        torch.save({
            'flow_state_dict': self.model.flow.state_dict(),
            'classifier_state_dict': self.model.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }, save_path)
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.flow.load_state_dict(save_dict['flow_state_dict'])
        self.model.classifer.load_state_dict(save_dict['classifier_state_dict'])

           
class AidedExtractorTrainer(ExtractorTrainer):
    def __init__(self, model, optimizer, train_loader, dev_loader, aided_loader, num_class=2, label_smoothe=0.):
        super(AidedExtractorTrainer, self).__init__(model, optimizer, train_loader, dev_loader, num_class, label_smoothe)
        self.aided_loader = aided_loader

    def aided_train(self):
        loss_meter = AverageMeter()
        for i, (x, label) in enumerate(self.aided_loader):
            x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
            self.optimizer.zero_grad()
            log_prob = self.model.flow.log_prob(x, label)
            loss = -torch.mean(log_prob)
            if torch.isnan(loss):
                print("nan")
                continue
            loss.backward()
            self.optimizer.step()
            loss_meter.update(loss.item())
            # if i % 20 == 0 :
            #     print(f"loss = {loss_meter.avg}")
        print(f"##aided loss : {loss_meter.avg}")
    
    def on_epoch_start(self):
        self.model.flow.train()
        self.aided_train()

    def save(self, save_path):
        torch.save({
            'flow_state_dict': self.model.flow.state_dict(),
            'classifier_state_dict': self.model.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'aided_loader': self.aided_loader,
            }, save_path)

        
