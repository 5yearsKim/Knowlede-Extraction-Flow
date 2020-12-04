import torch
from KEflow.model import prepare_flow
from KEflow.trainer import Inferencer
from KEflow.config import TYPE_FLOW
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import FLOW_CONFIG as Fcfg

# load model from checkpoint 
flow = prepare_flow(TYPE_FLOW, Ccfg["NC"], Ccfg["N_CLASS"])

state_dict = torch.load("ckpts/flow_best.pt")
flow.load_state_dict(state_dict["model_state"])

# inference
size = (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"])
inferencer = Inferencer(flow, size)
data, label = inferencer.inference(temp=.1)
# data = inferencer.amplify(data)

# save picture
inferencer.save_pic(data, "inference_sample/result.png")
print(label)