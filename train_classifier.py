import torch
from torch import nn
from dataloader import prepare_data
from KEflow.model import prepare_classifier
from KEflow.model.utils import weights_init
from KEflow.trainer import Trainer
from KEflow.config import TYPE_DATA, TYPE_CLS
from KEflow.config import CLS_CONFIG as Ccfg


""" dataloader """
trainset, devset = prepare_data("./data", TYPE_DATA)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Ccfg["BATCH_SIZE"])
devloader = torch.utils.data.DataLoader(devset, batch_size=Ccfg["BATCH_SIZE"])

""" define model """
model = prepare_classifier(TYPE_CLS, Ccfg["NC"], Ccfg["N_CLASS"])
# model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=Ccfg["LR"], weight_decay=Ccfg["WD"])

""" criterion define """
criterion = nn.CrossEntropyLoss()

""" train """
trainer = Trainer(model, optimizer, criterion, trainloader, devloader, best_save_path="ckpts/KEflow/")
# trainer.load("ckpts/KEflow/vgg11_bn.pt")
trainer.train(Ccfg["EPOCHS"], Ccfg["PRINT_FREQ"], Ccfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/KEflow/classifier.pt")

