import torch
from torch import nn
from dataloader import prepare_data
from KEflow.model import BasicCNN, LeNet5
from KEflow.trainer import Trainer
from KEflow.config import CLS_CONFIG as Ccfg


""" dataloader """
trainset, devset = prepare_data("./data", Ccfg["TYPE"])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Ccfg["BATCH_SIZE"])
devloader = torch.utils.data.DataLoader(devset, batch_size=Ccfg["BATCH_SIZE"])

""" define model """
model = LeNet5(nc=Ccfg["NC"], im_size=Ccfg["IM_SIZE"], n_filter=Ccfg["N_FILTER"], n_class=Ccfg["N_CLASS"])
optimizer = torch.optim.Adam(model.parameters(), lr=Ccfg["LR"], weight_decay=Ccfg["WD"])

""" criterion define """
criterion = nn.CrossEntropyLoss()

""" train """
trainer = Trainer(model, optimizer, criterion, trainloader, devloader, best_save_path="ckpts/KEflow/")
# trainer.load("ckpts/classifier.pt")
trainer.train(epochs=Ccfg["EPOCHS"], print_freq=Ccfg["PRINT_FREQ"], val_freq=Ccfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/KEflow/classifier.pt")

