import torch
import torchvision
from dataloader import prepare_data
from dataloader import PriorDataset
from KEflow.trainer.utils import dfs_freeze
from KEflow.model import prepare_classifier, AffineNICE, Glow, Extractor
from KEflow.trainer import AidedExtractorTrainer
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import TYPE_FLOW, TYPE_CLS, TYPE_DATA
if TYPE_FLOW == "NICE":
    from KEflow.config import NICE_CONFIG as Fcfg
elif TYPE_FLOW == "GLOW":
    from KEflow.config import GLOW_CONFIG as Fcfg
else:
    raise ValueError()

# define models / load classifier
if TYPE_FLOW == "NICE":
    flow = AffineNICE(Ccfg["NC"], Ccfg["IM_SIZE"], Fcfg["COUPLING"], Fcfg["COND_DIM"], \
                        Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
elif TYPE_FLOW == "GLOW":
    flow = Glow(Fcfg["IN_CHANNELS"], Fcfg["MID_CHANNELS"], Fcfg["COND_DIM"], \
                    Fcfg["NUM_LEVELS"], Fcfg["NUM_STEPS"] )

classifier = prepare_classifier(TYPE_CLS , Ccfg["NC"], Ccfg["N_CLASS"] )

state_dict = torch.load("ckpts/KEflow/classifier.pt")
classifier.load_state_dict(state_dict["model_state"])

extractor = Extractor(flow, classifier, Fcfg["ALPHA"], Fcfg["BETA"])

# freeze classifier part
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"], weight_decay=Fcfg["WD"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(500, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"])
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

_, aidedset = prepare_data('./data', TYPE_DATA)
aided_loader = torch.utils.data.DataLoader(aidedset, batch_size=Fcfg["AIDED_BATCH_SIZE"])


# train model
trainer = AidedExtractorTrainer(extractor, optimizer, train_loader, dev_loader, aided_loader, \
                                num_class=Fcfg["COND_DIM"], label_smoothe=Fcfg["SMOOTHE"], best_save_path="ckpts/KEflow")
# trainer.load("ckpts/extractor.pt")
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/extractor.pt")