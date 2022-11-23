import torch
from torch.utils.data import Dataset, DataLoader
import random
from .data_loader import PriorDataset
from .utils import to_one_hot
import json

class PseudoGenerator:
    def __init__(self, flow, teacher, data_size, cond_dim, temp=0.5, log_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow.to(self.device).eval()
        self.teacher = teacher.to(self.device).eval()
        self.data_size = data_size
        self.cond_dim = cond_dim
        self.temp = temp
        self.log_path = './log.json' if log_path is None else log_path

    def img_to_classifier(self, y):
        amp = random.uniform(1., 2. )
        return  amp * (y - 0.5)

    def generate(self, num_batch, batch_size):
        dset = PriorDataset(num_batch*batch_size, self.data_size, self.cond_dim, self.temp)
        noise_loader = DataLoader(dset, batch_size=batch_size ) 
        with open(self.log_path, 'w') as f:
            for i, (z, label) in enumerate(noise_loader):
                z, label = z.to(self.device), to_one_hot(label, self.cond_dim).to(self.device)
                x, _ = self.flow(z, label, reverse=True)
                x = self.img_to_classifier(x)
                label = self.teacher(x)
                label = torch.nn.functional.softmax(label, dim=-1)
                x, label = x.to('cpu').tolist(), label.to('cpu').tolist()
                for x_, label_ in zip(x, label):
                    json.dump([x_, label_], f)
                    f.write('\n')
                if i%10 == 0:
                    print(f'{i}th batch complete!')

class NoisePseudoGenerator:
    def __init__(self, teacher, data_size, cond_dim, temp, log_path=None, prior_type='normal'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = teacher.to(self.device).eval()
        self.data_size = data_size
        self.cond_dim = cond_dim
        self.temp = temp
        self.log_path = './log.json' if log_path is None else log_path
        self.prior_type = prior_type

    def generate(self, num_batch, batch_size):
        dset = PriorDataset(num_batch*batch_size, self.data_size, self.cond_dim, self.temp, prior_type=self.prior_type)
        noise_loader = DataLoader(dset, batch_size=batch_size)
        with open(self.log_path, 'w') as f:
            for i, (z, _) in enumerate(noise_loader):
                z = z.to(self.device)
                label  = self.teacher(z)
                label = torch.nn.functional.softmax(label, dim=1)
                z, label = z.to('cpu').tolist(), label.to('cpu').tolist()
                for z_, label_ in zip(z, label):
                    json.dump([z_, label_], f)
                    f.write('\n')
                if i%10 == 0:
                    print(f'{i}th batch complete!')


class KegnetPseudoGenerator:
    def __init__(self, kegnet, teacher, temp=1., log_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kegnet = kegnet.to(self.device).eval()
        self.teacher = teacher.to(self.device).eval()
        self.ny = kegnet.num_classes
        self.nz = kegnet.num_noises
        self.temp = temp
        self.log_path = './log.json' if log_path is None else log_path

    def generate(self, num_batch, batch_size):
        dset = PriorDataset(num_batch*batch_size, (self.nz,), self.ny, self.temp )
        noise_loader = DataLoader(dset, batch_size=batch_size)
        with open(self.log_path, 'w') as f:
            for i, (z, label) in enumerate(noise_loader):
                z, label = z.to(self.device), to_one_hot(label, self.ny).to(self.device)
                x = self.kegnet(label, z)
                label = self.teacher(x)
                label = torch.nn.functional.softmax(label, dim=-1)
                x, label = x.to('cpu').tolist(), label.to('cpu').tolist()
                for x_, label_ in zip(x, label):
                    json.dump([x_, label_], f)
                    f.write('\n')
                if i%10 == 0:
                    print(f'{i}th batch complete!')

