import torch
from torch import nn
from trainer import Trainer
from model import Classifier
from dataloader import ToyDataset
from config import *

model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

trainset = ToyDataset(NUM_SAMPLE)
devset = ToyDataset(100)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=BATCH_SIZE, shuffle=True)

trainer = Trainer(model, optimizer, criterion, train_loader, dev_loader)
trainer.train(EPOCHS, print_freq=PRINT_FREQ, val_freq=VAL_FREQ)