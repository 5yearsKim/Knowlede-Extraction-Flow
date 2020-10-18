import torch
from dataloader import PriorDataset
from KEflow.model import AffineNICE, Glow
from KEflow.trainer import Inferencer
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import TYPE
if TYPE == "NICE":
    from KEflow.config import NICE_CONFIG as Fcfg
elif TYPE == "GLOW":
    from KEflow.config import GLOW_CONFIG as Fcfg
else:
    raise ValueError()


# dataset for inference
dataset = PriorDataset(32, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"], temp=.5)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

# load model from checkpoint 
if TYPE == "NICE":
    flow = AffineNICE(Ccfg["NC"], Ccfg["IM_SIZE"], Fcfg["COUPLING"], Fcfg["COND_DIM"], \
                        Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
elif TYPE == "GLOW":
    flow = Glow(Fcfg["IN_CHANNELS"], Fcfg["MID_CHANNELS"], Fcfg["COND_DIM"], \
                    Fcfg["NUM_LEVELS"], Fcfg["NUM_STEPS"] )

state_dict = torch.load("ckpts/KEflow/extractor.pt")
flow.load_state_dict(state_dict["flow_state"])

# inference
inferencer = Inferencer(flow, dataloader)
data, label = inferencer.inference()
data = inferencer.amplify(data)

# save picture
inferencer.save_pic(data, "inference_sample/result.png")
print(label)