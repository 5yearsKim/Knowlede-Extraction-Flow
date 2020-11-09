import torchvision
from torchvision import transforms

def prepare_data(root, data_type, Normalize=True):
    if data_type == "DIGIT":
        transform =[
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
        if Normalize:
            transform.append(transforms.Normalize((0.1307,), (0.3081,)) )
        transform = transforms.Compose(transform)
        trainset = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
        devset = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)
    elif data_type == "FASHION":
        transform =[
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
        if Normalize:
            transform.append(transforms.Normalize( (0.2856,), (0.3385,)))
        transform = transforms.Compose(transform)
        trainset = torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        devset = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    elif data_type == "SVHN":
        stat = ((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))
        transform = [
            transforms.ToTensor(),
        ]
        if Normalize:
            transform.append(transforms.Normalize(*stat))   
        transform = transforms.Compose(transform)
        trainset = torchvision.datasets.SVHN(root, split='train', download=True, transform=transform)
        devset = torchvision.datasets.SVHN(root, split='test', download=True, transform=transform)
    elif data_type == "CIFAR":
        stat =  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform = [
            transforms.ToTensor(),
        ]
        if Normalize:
            transform.append(transforms.Normalize(*stat)) 
        transform = transforms.Compose(transform)
        
        trainset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
        devset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {data_type} not supported..")
    
    return trainset, devset