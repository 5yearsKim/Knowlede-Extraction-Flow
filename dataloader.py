from toy_distribution import mixed, moon2, moon1, circle
import torch

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, type='mixed'):
        self.num_samples = num_samples
        if type == 'moon2':
            data, label = moon2(num_samples)
        elif type == 'circle':
            data,label = circle(num_samples)
        elif type == 'mixed':
            data, label = mixed(num_samples)
        elif type == 'moon1':
            data, label = moon1(num_samples)
        else:
            raise ValueError(f'type {type} is not supported!')
        self.data, self.label = torch.from_numpy(data).to(torch.float32), torch.from_numpy(label)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

class PriorDataset(torch.utils.data.Dataset):
    def __init__(self, prior, in_out_dim, cond_dim, num_samples):
        self.num_samples = num_samples
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.cond_dim = cond_dim

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        data = self.prior.sample([self.in_out_dim])
        label = torch.randint(0, self.cond_dim, [1]).item()
        return data, label


if __name__ == "__main__":
    # dset = ToyDataset(100)
    # print(dset[0])
    prior = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))
    dset = PriorDataset(prior, 2, 2, 100)
    print(dset[0])
        