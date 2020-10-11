import torch
from model import Classifier, AffineNICE, Extractor
from trainer.utils import dfs_freeze
from trainer import ExtractorTrainer
from dataloader import PriorDataset
from config import FLOW_CONFIG as Fcfg
from config import CLS_CONFIG as Ccfg

# define prior distribution 
prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))

# define models / load classifier
flow = AffineNICE(prior, Fcfg["COUPLING"], Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
classifier = Classifier(Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["N_FILTER"] )

state_dict = torch.load("ckpts/classifier.pt")
classifier.load_state_dict(state_dict["model_state_dict"])

extractor = Extractor(flow, classifier)

# freeze classifier part
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"])

# dataloader setting
trainset = PriorDataset(prior, Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(prior, 200, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)

# train model
trainer = ExtractorTrainer(extractor, optimizer, train_loader, dev_loader, num_class=Fcfg["COND_DIM"], label_smoothe=Fcfg["SMOOTHE"])
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/extractor.pt")