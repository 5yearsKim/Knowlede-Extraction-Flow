import torch
from trainer import Trainer
from model import Classifier
from dataloader import ToyDataset
from config import *

net = Classifier()

trainset = ToyDataset(NUM_SAMPLE)
devset = ToyDataset(100)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=BATCH_SIZE)

trainer = Trainer(net, train_loader, dev_loader)
trainer.train()