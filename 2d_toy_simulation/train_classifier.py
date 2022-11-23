import torch
from torch import nn
from trainer import Trainer
from model import Classifier
from dataloader import ToyDataset
from config import CLS_CONFIG as cfg
from trainer.utils import LabelSmoothLoss


"""  model define """
model = Classifier(cfg["DIM_IN"], cfg["DIM_OUT"], cfg["N_HIDDEN"], cfg["DIM_HIDDEN"])
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])

""" criterion define """
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothLoss(smoothing=0.1) 


""" dataloader  """
trainset = ToyDataset(cfg["NUM_SAMPLE"], type=cfg["TOY_TYPE"])
devset = ToyDataset(1000, type=cfg["TOY_TYPE"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=cfg["BATCH_SIZE"], shuffle=True)

""" train """
trainer = Trainer(model, optimizer, criterion, train_loader, dev_loader)
trainer.train(cfg["EPOCHS"], print_freq=cfg["PRINT_FREQ"], val_freq=cfg["VAL_FREQ"])

""" save model """
trainer.save("ckpts/classifier.pt")