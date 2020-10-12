import torch
from model import AffineNICE
from trainer import Inferencer
from dataloader import PriorDataset
from config import FLOW_CONFIG as Fcfg
from config import CLS_CONFIG as Ccfg

prior = torch.distributions.Normal(0, 1)
# dataset for inference
dataset = PriorDataset(prior, 32, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"], temp=1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)


# load model from checkpoint 
flow = AffineNICE(prior, Fcfg["COUPLING"], Fcfg["IN_OUT_DIM"], Fcfg["COND_DIM"], Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
state_dict = torch.load("ckpts/extractor.pt")
flow.load_state_dict(state_dict["flow_state_dict"])

# inference
inferencer = Inferencer(flow, dataloader)
data, label = inferencer.inference()

# save picture
inferencer.save_pic(data, "inference_sample/result.png")
print(label)