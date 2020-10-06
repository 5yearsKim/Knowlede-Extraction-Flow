import torch
from model import Classifier, AffineNICE, Extractor
from trainer.utils import dfs_freeze
from trainer import AidedExtractorTrainer
from dataloader import PriorDataset, ToyDataset
from config import FLOW_CONFIG as Fcfg
from config import CLS_CONFIG as Ccfg

prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))

# define models / load classifier
flow = AffineNICE(prior, Fcfg["COUPLING"], Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
classifier = Classifier(Ccfg["DIM_IN"], Ccfg["DIM_OUT"], Ccfg["N_HIDDEN"], Ccfg["DIM_HIDDEN"])

state_dict = torch.load("ckpts/classifier.pt")
classifier.load_state_dict(state_dict["model_state_dict"])

extractor = Extractor(flow, classifier)

# freeze classifier part
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"], weight_decay=Fcfg["WD"])

# dataloader setting
trainset = PriorDataset(prior, Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["NUM_SAMPLE"])
devset = PriorDataset(prior, Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], 100)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

aidedset = ToyDataset(Fcfg["NUM_AIDED_SAMPLE"], type=Ccfg["TOY_TYPE"])
aidedloader = torch.utils.data.DataLoader(aidedset, batch_size=Fcfg["AIDED_BATCH_SIZE"], shuffle=True)

# train model
trainer = AidedExtractorTrainer(extractor, optimizer, train_loader, dev_loader, aidedloader, num_class=Fcfg["COND_DIM"])
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/extractor.pt")