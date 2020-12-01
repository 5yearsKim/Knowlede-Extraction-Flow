import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class PriorDataset(Dataset):
    def __init__(self, num_sample, data_size, cond_dim, temp=1.):
        self.prior = torch.distributions.Normal(0, 1)
        self.num_sample = num_sample
        self.data_size = data_size
        self.cond_dim = cond_dim
        self.temp = temp

    def __len__(self):
        return self.num_sample
    
    def __getitem__(self, index):
        data = self.prior.sample(self.data_size) *self.temp
        label = torch.randint(0, self.cond_dim, [1]).item()
        return data, label


class SelectedDataset(Dataset):
    def __init__(self, root, dset_type="DIGIT"):
        self.root = root
        self.dset_type = dset_type
        self.files = os.listdir(root)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
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
        return image, label


if __name__ == "__main__":
    dataset = SelectedDataset("aided_sample/DIGIT_selected")
    print(dataset[1])