import torch
from data_handler import PriorDataset, SelectedDataset
from KEflow.trainer import ExtractorTrainer, AidedExtractorTrainer
from KEflow.model import prepare_classifier, prepare_flow
from config import TYPE_DATA, DISTRIBUTION, im_size, cls_type, flow_lr, flow_bs, aided_weight, flow_epochs, flow_print_freq, det_s, bn_s
import random

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

is_aided = True if DISTRIBUTION == "KEFLOW+" else False
print(f"is aided = {is_aided}")

# define models / load classifier
flow = prepare_flow('NICE', im_size[0] , 10)
classifier = prepare_classifier(cls_type, im_size[0], 10)

# load teacher network to extract knowledge
state_dict = torch.load(f"ckpts/teacher/classifier_{TYPE_DATA.lower()}_{cls_type.lower()}_.pt")
classifier.load_state_dict(state_dict["model_state"])

# optimizer
optimizer = torch.optim.Adam(flow.parameters(), flow_lr, weight_decay=1e-3)

# dataloader setting
trainset = PriorDataset(20000, im_size, 10, temp =.5)
devset = PriorDataset(500, im_size, 10, temp=.5)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=flow_bs, shuffle=True)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=flow_bs)

if is_aided == False:
    # train model
    trainer = ExtractorTrainer(classifier, flow, optimizer, train_loader, dev_loader,\
                                num_class=10, best_save_path="ckpts/generator/")

    trainer.train(flow_epochs, flow_print_freq, val_freq=1, spread_s=det_s, gravity_s=0, bn_s=bn_s)

    # save model
    trainer.save(f"ckpts/generator/keflow_{TYPE_DATA.lower()}.pt")
else :
    aidedset = SelectedDataset(f'data/aided_sample/{TYPE_DATA}_selected', TYPE_DATA, one_hot=False)
    aided_loader = torch.utils.data.DataLoader(aidedset, batch_size=10)

    # train model
    trainer = AidedExtractorTrainer(classifier, flow, optimizer, train_loader, dev_loader, aided_loader, \
                                    num_class=10, aided_weight=aided_weight, best_save_path="ckpts/generator/")
    # trainer.load("ckpts/flow_best.pt")

    trainer.train(flow_epochs, flow_print_freq, val_freq=1, spread_s=det_s, gravity_s=0, bn_s=bn_s)

    # # save model
    trainer.save(f"ckpts/generator/aided_keflow_{TYPE_DATA.lower()}.pt")
