import torch
from dataloader import PriorDataset
from KEflow.trainer import ExtractorTrainer
from KEflow.model import prepare_classifier, prepare_flow
from KEflow.config import TYPE_FLOW, TYPE_CLS, TYPE_DATA
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import FLOW_CONFIG as Fcfg


# define models / load classifier
flow = prepare_flow(TYPE_FLOW, Ccfg["NC"], Ccfg["N_CLASS"])
classifier = prepare_classifier(TYPE_CLS, Ccfg["NC"], Ccfg["N_CLASS"])

state_dict = torch.load(f"ckpts/classifier_{TYPE_DATA.lower()}_{TYPE_CLS.lower()}.pt")
# state_dict = torch.load(f"kegnet_pretrained/{TYPE_DATA.lower()}.pth.tar", map_location='cuda:0')
classifier.load_state_dict(state_dict["model_state"])

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), Fcfg["LR"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"], temp = .5)
devset = PriorDataset(500, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"], temp=.5)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

# train model
trainer = ExtractorTrainer(classifier, flow, optimizer, train_loader, dev_loader,\
                            num_class=Ccfg["N_CLASS"], best_save_path="ckpts")

# trainer.load("ckpts/flow_best.pt")
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"], Fcfg["SPREAD_S"], Fcfg["GRAVITY_S"], Fcfg["BN_S"])

# # save model
# trainer.save("ckpts/flow.pt")
