import torch
from torch import nn
import torchvision
from torchvision import transforms
from KEflow.model import BasicCNN, LeNet5
from KEflow.trainer import Trainer
from KEflow.config import CLS_CONFIG as Ccfg


""" dataloader """
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

if Ccfg["TYPE"] == "DIGIT":
    trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    devset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
elif Ccfg["TYPE"] == "FASHION":
    trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    devset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=Ccfg["BATCH_SIZE"])
devloader = torch.utils.data.DataLoader(devset, batch_size=Ccfg["BATCH_SIZE"])

""" define model """
model = LeNet5(nc=Ccfg["NC"], im_size=Ccfg["IM_SIZE"], n_filter=Ccfg["N_FILTER"], n_class=Ccfg["N_CLASS"])
optimizer = torch.optim.Adam(model.parameters(), lr=Ccfg["LR"], weight_decay=Ccfg["WD"])

""" criterion define """
criterion = nn.CrossEntropyLoss()

""" train """
trainer = Trainer(model, optimizer, criterion, trainloader, devloader)
# trainer.load("ckpts/classifier.pt")
trainer.train(epochs=Ccfg["EPOCHS"], print_freq=Ccfg["PRINT_FREQ"], val_freq=Ccfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/KEflow/classifier.pt")

