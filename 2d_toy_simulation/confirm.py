import torch
from model import Classifier, NICE 
from dataloader import PriorDataset
from config import FLOW_CONFIG as Fcfg
from config import CLS_CONFIG as Ccfg
from utils import to_one_hot
import matplotlib.pyplot as plt
import numpy as np

# dataset for inference
prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))

dataset = PriorDataset(prior, Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], 10 )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# load model from checkpoint 
flow = NICE(prior, Fcfg["COUPLING"], Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
state_dict = torch.load("ckpts/extractor.pt")
flow.load_state_dict(state_dict["flow_state_dict"])

classifier = Classifier(Ccfg["DIM_IN"], Ccfg["DIM_OUT"], Ccfg["N_HIDDEN"], Ccfg["DIM_HIDDEN"])
classifier.load_state_dict(state_dict["classifier_state_dict"])
# state_dict = torch.load("ckpts/classifier.pt")
# classifier.load_state_dict(state_dict["model_state_dict"])


classifier.eval()

softmax = torch.nn.Softmax(dim=1)

for z, label in dataloader:
    cond = to_one_hot(label, Fcfg["COND_DIM"])        
    out, _ = flow(z, cond, reverse=True)
    logit = classifier(out)
    print("loc = ", out.detach())
    print(softmax(logit.detach()), label, '\n')

# with torch.no_grad():
#     for i in range(10):
#         loc = prior.sample((1,2)) * 5
#         logit = classifier(loc)
#         print(loc.square().sum().sqrt())
#         # print(loc)
#         print(logit.detach())
#         print(softmax(logit.detach()), "\n")

  
