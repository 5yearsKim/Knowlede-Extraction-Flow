import torch
from model import AffineNICE
from inferencer import Inferencer
from dataloader import PriorDataset
from config import FLOW_CONFIG as Fcfg
import matplotlib.pyplot as plt
import numpy as np

# dataset for inference
prior = torch.distributions.Normal( torch.tensor(0.), torch.tensor(1.))

dataset = PriorDataset(prior, Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], 4000 )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)

# load model from checkpoint 
flow = AffineNICE(prior, Fcfg["COUPLING"], Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
state_dict = torch.load("ckpts/extractor.pt")
flow.load_state_dict(state_dict["flow_state_dict"])

# inference
inferencer = Inferencer(flow, dataloader, cond_dim=Fcfg["COND_DIM"])
data, label = inferencer.inference()

# drawing graph
data, label = data.detach().numpy(), label.detach().numpy()
colormap = np.array(['b', 'r'])
plt.scatter(data[:, 0], data[:, 1], c=colormap[label], s=1)
plt.show()