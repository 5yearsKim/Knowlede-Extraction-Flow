import torch
from torch import nn
import torchvision
from torchvision import transforms
from model import Classifier
from trainer import Trainer
from config import CLS_CONFIG as Ccfg


""" dataloader """
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
devset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=Ccfg["BATCH_SIZE"])
devloader = torch.utils.data.DataLoader(devset, batch_size=Ccfg["BATCH_SIZE"])

""" define model """
model = Classifier(nc=Ccfg["NC"], im_size=Ccfg["IM_SIZE"], n_filter=Ccfg["N_FILTER"], n_class=Ccfg["N_CLASS"])
optimizer = torch.optim.Adam(model.parameters(), lr=Ccfg["LR"], weight_decay=Ccfg["WD"])

""" criterion define """
criterion = nn.CrossEntropyLoss()

""" train """
trainer = Trainer(model, optimizer, criterion, trainloader, devloader)
trainer.train(epochs=Ccfg["EPOCHS"], print_freq=Ccfg["PRINT_FREQ"], val_freq=Ccfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/classifier.pt")

