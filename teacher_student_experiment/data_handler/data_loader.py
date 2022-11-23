import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import os
from .utils import to_one_hot



class PriorDataset(Dataset):
    def __init__(self, num_sample, data_size, cond_dim, temp=1., prior_type='normal', one_hot=False):
        if prior_type == 'normal':
            self.prior = torch.distributions.Normal(0, 1) 
        elif prior_type == 'uniform':
            self.prior = torch.distributions.Uniform(-0.5, 0.5)
        self.num_sample = num_sample
        self.data_size = data_size
        self.cond_dim = cond_dim
        self.temp = temp
        self.one_hot = one_hot

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        data = self.prior.sample(self.data_size) *self.temp
        label = torch.randint(0, self.cond_dim, [1]).item()
        if self.one_hot:
            label = to_one_hot(label, self.cond_dim)
        return data, label


class PseudoDataset(Dataset):
    def __init__(self, data_path):
        if isinstance(data_path, str):
            data_path = [data_path]
        self.data = []
        for path in data_path:
            with open(path, 'r') as f:
                for line in f:
                    if line == "":
                        continue
                    self.data.append(json.loads(line.rstrip('\n|\r')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img, label = torch.tensor(img), torch.tensor(label)
        return img, label


class SelectedDataset(Dataset):
    def __init__(self, root, dset_type="DIGIT", one_hot=True, repeat=1):
        self.root = root
        self.dset_type = dset_type
        self.files = os.listdir(root)
        self.one_hot = one_hot
        self.repeat = int(repeat)

    def __len__(self):
        return len(self.files) * self.repeat
    
    def __getitem__(self, index):
        if self.repeat > 1:
            index = index % len(self.files)
        file_path = os.path.join(self.root, self.files[index])
        label = int(file_path[-5])
        image = Image.open(file_path)
        if self.dset_type in ["DIGIT", "FASHION"]:
            m_tr = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            m_tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        image = m_tr(image)
        if self.one_hot:
            label = to_one_hot(label, 10)
        return image, label
