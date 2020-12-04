import torch
from torch import nn
import torch.nn.functional as F
import os
import random
from .utils import to_one_hot, AverageMeter, DeepInversionFeatureHook, label_smoothe, img_reg_loss

class ExtractorTrainer:
    def __init__(self, classifier, flow, optimizer, train_loader, dev_loader, num_class=10, best_save_path="ckpts/"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = classifier.to(self.device).eval()
        self.flow = flow.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.val_best = 0.
        self.num_class = num_class
        self.best_save_path = best_save_path
        self.first_bn_multiplier = 5.
        self.bn_feat_layers = []
        for module in classifier.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_feat_layers.append(DeepInversionFeatureHook(module))

        self.ll_loss_meter = AverageMeter()
        self.ldj_loss_meter = AverageMeter()
        self.grav_loss_meter = AverageMeter()
        self.bn_loss_meter = AverageMeter()

    def train(self, epochs, print_freq=10, val_freq=1, spread_s=0.04, gravity_s=1, bn_s=1. ,label_smoothe=0.):
        loss_meter = AverageMeter()
        for epoch in range(epochs):
            self.on_every_epoch()
            self.flow.train()
            loss_meter.reset()
            self.ll_loss_meter.reset()
            self.ldj_loss_meter.reset()
            self.grav_loss_meter.reset()
            self.bn_loss_meter.reset() 
            for i, (x, label) in enumerate(self.train_loader):
                self.train_step(x, label, loss_meter, spread_s, gravity_s, bn_s, label_smoothe)
                if i%print_freq == 0:
                    print(f'iter {i} : loss = {loss_meter.avg:.3f}| ll:{self.ll_loss_meter.avg:.3f}, ldj:{self.ldj_loss_meter.avg:.3f}, grav:{self.grav_loss_meter.avg:.3f}, bn:{self.bn_loss_meter.avg:.3f} ')
            print(f'*epoch{epoch} : loss = {loss_meter.avg:.3f}| ll:{self.ll_loss_meter.avg:.3f}, ldj:{self.ldj_loss_meter.avg:.3f}, grav:{self.grav_loss_meter.avg:.3f}, bn:{self.bn_loss_meter.avg:.3f} ')
            
            if i%val_freq ==0:
                with torch.no_grad():
                    val_best = self.validate(epoch)


    def train_step(self, x, label, loss_meter, spread_s, gravity_s, bn_s, smoothing):
        x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
        self.optimizer.zero_grad()

        y, log_det_J = self.flow(x, label, reverse=True) 
        reg = img_reg_loss(y) #regularization loss
        y = 2*y -1.

        lim = 2
        # apply random jitter offsets
        off1 = random.randint(-lim, lim)
        off2 = random.randint(-lim, lim)
        y = torch.roll(y, shifts=(off1,off2), dims=(2,3))

        confidence = F.softmax(self.classifier(y), dim=1)
        # confidence = torch.clamp(confidence, 0., 0.7)
        confidence = torch.log(confidence)
        if smoothing != 0.:
            label = label_smoothe(label, smoothing)
        bs, class_dim = label.size(0), label.size(1)
        log_ll = torch.bmm(confidence.view(bs, 1, class_dim), label.view(bs, class_dim, 1)).view(bs)
        
        if self.bn_feat_layers:
            rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.bn_feat_layers)-1)]
            bn_loss = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_feat_layers)])
        else:
            bn_loss = 0.
        
        ll_loss = - torch.mean(log_ll)
        ldj_loss = - spread_s * torch.mean(log_det_J)
        grav_loss = gravity_s * torch.mean(reg)
        bn_loss = bn_s * bn_loss
        
        loss = ll_loss + ldj_loss + grav_loss + bn_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1)
        self.optimizer.step()
        
        loss_meter.update(loss.item())
        self.ll_loss_meter.update(ll_loss.item())
        self.ldj_loss_meter.update(ldj_loss.item())
        self.grav_loss_meter.update(grav_loss.item())
        self.bn_loss_meter.update(bn_loss.item())

    @torch.no_grad()
    def validate(self, epoch):
        self.flow.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for x, label in self.dev_loader:
                x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
                acc = self.get_acc(x, label)
                acc_meter.update(acc, x.size(0))
            print(f"[{epoch} epoch Validation]: acc : {acc_meter.avg}")
        if acc_meter.avg > 0:#self.val_best:
            self.val_best = acc_meter.avg
            path = os.path.join(self.best_save_path, "flow_best.pt")
            self.save(path) 
            print("saving BEST..")

    def get_acc(self, x, cond):
        y, log_det_J = self.flow(x, cond, reverse=True) 
        y = 2 * y - 1.
        _, predicted = torch.max(self.classifier(y), 1)
        _, cond = torch.max(cond, 1)
        hit_rate = float(predicted.eq(cond).sum())/ float(cond.size(0))
        return hit_rate


    def save(self, save_path):
        torch.save({
            'model_state': self.flow.state_dict(),
            }, save_path)
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.flow.load_state_dict(save_dict['model_state'])

    def on_every_epoch(self):
        pass

           
class AidedExtractorTrainer(ExtractorTrainer):
    def __init__(self, classifier, flow, optimizer, train_loader, dev_loader, aided_loader, num_class=2, aided_weight=1., best_save_path="ckpts"):
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x
        super(AidedExtractorTrainer, self).__init__(classifier, flow, optimizer, train_loader, dev_loader, num_class, best_save_path)
        self.aided_loader = iter(cycle(aided_loader))
        self.aided_meter = AverageMeter()
        self.aided_weight = aided_weight

    def train_step(self, x, label, loss_meter, spread_s, gravity_s, bn_s, smoothing):
        x, label = x.to(self.device), to_one_hot(label, self.num_class).to(self.device)
        self.optimizer.zero_grad()

        y, log_det_J = self.flow(x, label, reverse=True) 
        reg = img_reg_loss(y) #regularization loss
        y = 2*y -1

        lim = 2
        # apply random jitter offsets
        off1 = random.randint(-lim, lim)
        off2 = random.randint(-lim, lim)
        y = torch.roll(y, shifts=(off1,off2), dims=(2,3))

        confidence = F.softmax(self.classifier(y), dim=1)
        confidence = torch.log(confidence)
        if smoothing != 0.:
            label = label_smoothe(label, smoothing)
        bs, class_dim = label.size(0), label.size(1)
        log_ll = torch.bmm(confidence.view(bs, 1, class_dim), label.view(bs, class_dim, 1)).view(bs)
        
        if self.bn_feat_layers:
            rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.bn_feat_layers)-1)]
            bn_loss = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.bn_feat_layers)])
        else:
            bn_loss = 0.
        loss = torch.mean(-log_ll - spread_s * log_det_J + gravity_s * reg) + bn_s * bn_loss
        
        real_img, label = next(self.aided_loader)
        real_img, label = real_img.to(self.device), to_one_hot(label, self.num_class).to(self.device) 

        aided_loss = -self.aided_weight * torch.mean(self.flow.log_prob(real_img, label)) 
        loss += aided_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 1)
        self.optimizer.step()
        loss_meter.update(loss.item())
        self.aided_meter.update(aided_loss.item())

    def on_every_epoch(self):
        print(f"aided_loss : {self.aided_meter.avg}\n")
        self.aided_meter.reset()
