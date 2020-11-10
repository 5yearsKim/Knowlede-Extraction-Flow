import torch
import torchvision
from dataloader import prepare_data
from dataloader import PriorDataset
from KEflow.trainer.utils import dfs_freeze
from KEflow.model import prepare_classifier, prepare_flow, Extractor
from KEflow.trainer import AidedExtractorTrainer
from KEflow.config import TYPE_FLOW, TYPE_CLS, TYPE_DATA
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import FLOW_CONFIG as Fcfg

# define models / load classifier
flow = prepare_flow(TYPE_FLOW, Ccfg["NC"], Ccfg["N_CLASS"])
classifier = prepare_classifier(TYPE_CLS , Ccfg["NC"], Ccfg["N_CLASS"] )

state_dict = torch.load("ckpts/KEflow/classifier.pt")
# state_dict = torch.load("ckpts/from_kegnet/mnivst.pth.tar")
classifier.load_state_dict(state_dict["model_state"])

extractor = Extractor(flow, classifier, Fcfg["ALPHA"], Fcfg["BETA"])

# freeze classifier part
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"], weight_decay=Fcfg["WD"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(200, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"])
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

aidedset, _ = prepare_data('./data', TYPE_DATA, Normalize=False)
print(len(aidedset))
aidedset, _ = torch.utils.data.random_split(aidedset, [Fcfg["NUM_AIDED_SAMPLE"], len(aidedset) - Fcfg["NUM_AIDED_SAMPLE"]])
aided_loader = torch.utils.data.DataLoader(aidedset, batch_size=Fcfg["AIDED_BATCH_SIZE"])


# train model
trainer = AidedExtractorTrainer(extractor, optimizer, train_loader, dev_loader, aided_loader, \
                                num_class=Ccfg["N_CLASS"], aided_weight=200, label_smoothe=Fcfg["SMOOTHE"], best_save_path="ckpts/KEflow")
# trainer.load("ckpts/KEflow/extractor.pt")

# trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])
trainer.loss_sum_train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/KEflow/extractor.pt")