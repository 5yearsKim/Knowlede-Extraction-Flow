from toy_distribution import mixed
import torch

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, type='mixed'):
        self.num_samples = num_samples
        if type == 'moon2':
            data, label = moon2(num_samples)
        elif type == 'mixed':
            data, label = mixed(num_samples)
        else:
            raise ValueError(f'type {type} is not supported!')
        self.data, self.label = torch.from_numpy(data).to(torch.float32), torch.from_numpy(label)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

if __name__ == "__main__":
    dset = ToyDataset(100)
    print(dset[0])
        