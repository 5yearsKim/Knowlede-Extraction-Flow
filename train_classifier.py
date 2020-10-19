import torch
from torch import nn
from dataloader import prepare_data
from KEflow.model import BasicCNN, LeNet5, ResNet
from KEflow.trainer import Trainer
from KEflow.config import CLS_CONFIG as Ccfg


""" dataloader """
trainset, devset = prepare_data("./data", Ccfg["TYPE"])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Ccfg["BATCH_SIZE"])
devloader = torch.utils.data.DataLoader(devset, batch_size=Ccfg["BATCH_SIZE"])

""" define model """
model = ResNet(Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["N_FILTER"], Ccfg["N_CLASS"])
optimizer = torch.optim.Adam(model.parameters(), lr=Ccfg["LR"], weight_decay=Ccfg["WD"])

""" criterion define """
criterion = nn.CrossEntropyLoss()

""" train """
trainer = Trainer(model, optimizer, criterion, trainloader, devloader, best_save_path="ckpts/KEflow/")
# trainer.load("ckpts/classifier.pt")
trainer.train(Ccfg["EPOCHS"], Ccfg["PRINT_FREQ"], Ccfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/KEflow/classifier.pt")

