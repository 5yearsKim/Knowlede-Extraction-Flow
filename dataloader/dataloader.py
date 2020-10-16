import torch


class PriorDataset(torch.utils.data.Dataset):
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


if __name__ == "__main__":
    # dset = ToyDataset(100)
    # print(dset[0])
    pass        