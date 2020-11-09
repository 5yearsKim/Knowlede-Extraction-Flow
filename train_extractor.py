import torch
from dataloader import PriorDataset
from KEflow.trainer.utils import dfs_freeze
from KEflow.trainer import ExtractorTrainer
from KEflow.model import prepare_classifier, prepare_flow, Extractor
from KEflow.model.utils import weights_init
from KEflow.config import TYPE_FLOW, TYPE_CLS
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import FLOW_CONFIG as Fcfg


# define models / load classifier
flow = prepare_flow(TYPE_FLOW, Ccfg["NC"], Ccfg["N_CLASS"])
classifier = prepare_classifier(TYPE_CLS, Ccfg["NC"], Ccfg["N_CLASS"])

state_dict = torch.load("ckpts/KEflow/lenet_digit.pt")
# state_dict = torch.load("ckpts/from_kegnet/mnist.pth.tar")
classifier.load_state_dict(state_dict["model_state"])

extractor = Extractor(flow, classifier, Fcfg["ALPHA"], Fcfg["BETA"])

# freeze classifier part    
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(500, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

# train model
trainer = ExtractorTrainer(extractor, optimizer, train_loader, dev_loader,\
                             num_class=Ccfg["N_CLASS"], label_smoothe=Fcfg["SMOOTHE"], best_save_path="ckpts/KEflow")

# trainer.load("ckpts/KEflow/best.pt")
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/KEflow/extractor.pt")