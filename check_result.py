import torch
from KEflow.model import AffineNICE, Glow
from KEflow.trainer import Inferencer
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import TYPE_FLOW
if TYPE_FLOW == "NICE":
    from KEflow.config import NICE_CONFIG as Fcfg
elif TYPE_FLOW == "GLOW":
    from KEflow.config import GLOW_CONFIG as Fcfg
else:
    raise ValueError()

# load model from checkpoint 
if TYPE_FLOW == "NICE":
    flow = AffineNICE(Ccfg["NC"], Ccfg["IM_SIZE"], Fcfg["COUPLING"], Fcfg["COND_DIM"], \
                        Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
elif TYPE_FLOW == "GLOW":
    flow = Glow(Fcfg["IN_CHANNELS"], Fcfg["MID_CHANNELS"], Fcfg["COND_DIM"], \
                    Fcfg["NUM_LEVELS"], Fcfg["NUM_STEPS"] )

state_dict = torch.load("ckpts/KEflow/best.pt")
flow.load_state_dict(state_dict["flow_state"])

# inference
size = (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"])
inferencer = Inferencer(flow, size)
data, label = inferencer.inference(temp=0.01)
data = inferencer.amplify(data)

# save picture
inferencer.save_pic(data, "inference_sample/result.png")
print(label)