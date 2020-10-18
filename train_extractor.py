import torch
from dataloader import PriorDataset
from KEflow.trainer.utils import dfs_freeze
from KEflow.trainer import ExtractorTrainer
from KEflow.model import BasicCNN, LeNet5, AffineNICE, Glow, Extractor
from KEflow.config import CLS_CONFIG as Ccfg
from KEflow.config import TYPE
if TYPE == "NICE":
    from KEflow.config import NICE_CONFIG as Fcfg
elif TYPE == "GLOW":
    from KEflow.config import GLOW_CONFIG as Fcfg
else:
    raise ValueError()

# define models / load classifier
if TYPE == "NICE":
    flow = AffineNICE(Ccfg["NC"], Ccfg["IM_SIZE"], Fcfg["COUPLING"], Fcfg["COND_DIM"], \
                        Fcfg["MID_DIM"], Fcfg["HIDDEN"] )
elif TYPE == "GLOW":
    flow = Glow(Fcfg["IN_CHANNELS"], Fcfg["MID_CHANNELS"], Fcfg["COND_DIM"], \
                    Fcfg["NUM_LEVELS"], Fcfg["NUM_STEPS"] )

classifier = LeNet5(Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["N_FILTER"] )

state_dict = torch.load("ckpts/KEflow/classifier.pt")
# state_dict = torch.load("ckpts/from_kegnet/mnist.pth.tar")
classifier.load_state_dict(state_dict["model_state"])

extractor = Extractor(flow, classifier, Fcfg["ALPHA"], Fcfg["BETA"])

# freeze classifier part
dfs_freeze(extractor.classifier)

# optimizer
optimizer = torch.optim.Adam(extractor.flow.parameters(), Fcfg["LR"])

# dataloader setting
trainset = PriorDataset(Fcfg["NUM_SAMPLE"], (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])
devset = PriorDataset(200, (Ccfg["NC"], Ccfg["IM_SIZE"], Ccfg["IM_SIZE"]), Ccfg["N_CLASS"])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=Fcfg["BATCH_SIZE"], shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=Fcfg["BATCH_SIZE"])

# train model
trainer = ExtractorTrainer(extractor, optimizer, train_loader, dev_loader,\
                             num_class=Fcfg["COND_DIM"], label_smoothe=Fcfg["SMOOTHE"], best_save_path="ckpts/KEflow")

# trainer.load("ckpts/extractor.pt")
trainer.train(Fcfg["EPOCHS"], Fcfg["PRINT_FREQ"], Fcfg["VAL_FREQ"])

# save model
trainer.save("ckpts/KEflow/extractor.pt")