import torchvision


def prepare_data(data_type):
    if Ccfg["TYPE"] == "DIGIT":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        devset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif Ccfg["TYPE"] == "FASHION":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(((0.2856,), (0.3385,))
        ])
        trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        devset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    elif Ccfg["TYPE"] == "SVHN":
        stat = ((0.4378, 0.4439, 0.4729), (0.1980, 0.2011, 0.1971))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stat)
        ])       
        trainset = torchvision.datasets.FashionMNIST('./data', split='train', download=True, transform=transform)
        devset = torchvision.datasets.FashionMNIST('./data', split='test', download=True, transform=transform)
    elif Ccfg["TYPE"] == "CIFAR":
        stat =  ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(((0.2856,), (0.3385,))
        ])
        trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        devset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {data_type} not supported..")
