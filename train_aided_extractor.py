import torch
import torchvision
from dataloader import PriorDataset, SelectedDataset
from KEflow.model import prepare_classifier, prepare_flow
from KEflow.trainer import AidedExtractorTrainer
from KEflow.config import TYPE_FLOW, TYPE_CLS, TYPE_DATA
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import FLOW_CONFIG as Fcfg

# define models / load classifier
flow = prepare_flow(TYPE_FLOW, Ccfg["NC"], Ccfg["N_CLASS"])
classifier = prepare_classifier(TYPE_CLS , Ccfg["NC"], Ccfg["N_CLASS"] )

state_dict = torch.load("ckpts/classifier.pt")
# state_dict = torch.load("ckpts/from_kegnet/mnivst.pth.tar")
classifier.load_state_dict(state_dict["model_state"])

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), Fcfg["LR"], weight_decay=Fcfg["WD"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(200, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"])
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

aidedset = SelectedDataset(f'./aided_sample/{TYPE_DATA}_selected', TYPE_DATA)
aided_loader = torch.utils.data.DataLoader(aidedset, batch_size=10)


# train model
trainer = AidedExtractorTrainer(classifier, flow, optimizer, train_loader, dev_loader, aided_loader, \
                                num_class=Ccfg["N_CLASS"], aided_weight=.1, best_save_path="ckpts/")
# trainer.load("ckpts/best.pt")

trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"],  Fcfg["SPREAD_S"], Fcfg["GRAVITY_S"], Fcfg["BN_S"])

# save model
trainer.save("ckpts/extractor.pt")