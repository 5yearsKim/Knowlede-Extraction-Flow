import torch
import torchvision
from torchvision import transforms

def prepare_data(root, data_type):
    if data_type == "DIGIT":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        trainset = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
        devset = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
    elif data_type == "FASHION":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize( (0.2856,), (0.3385,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        devset = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    elif data_type == "SVHN":
        stat = ((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])
        trainset = torchvision.datasets.SVHN(root, split='train', download=True, transform=transform_train)
        devset = torchvision.datasets.SVHN(root, split='test', download=True, transform=transform_test)
    elif data_type == "CIFAR":
        stat =  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])       
        trainset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        devset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {data_type} not supported..")
    
    return trainset, devset


def normalize_image(x, dtype='DIGIT'):
    if dtype == 'DIGIT':
        stat = ((0.1307,), (0.3081,)) 
    elif dtype == 'FASHION':
        stat = ( (0.2856,), (0.3385,))
    elif dtype == 'SVHN' : 
        stat =  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dtype == 'CIFAR':
        stat =  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        raise ValueError()
    device = x.device
    mean, std = stat
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(device)
    return (x - mean)/std