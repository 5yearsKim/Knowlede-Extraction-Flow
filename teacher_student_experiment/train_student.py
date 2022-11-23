import sys
import torch
from data_handler import PseudoDataset, SelectedDataset, prepare_data
from config import TYPE_DATA, DISTRIBUTION, cls_type, im_size, stud_bs, stud_lr, stud_epochs, compress_type
from trainer import Trainer, KLDivLoss
from KEflow.model import prepare_classifier
from tucker.utils import count_parameters
from utils import Unbuffered
import wandb

use_wandb = True
if use_wandb:
    wandb.init(project=f'{TYPE_DATA.lower()}_student', name=f'{TYPE_DATA.lower()}_{DISTRIBUTION.lower()}_{compress_type}')
    config = wandb.config
    config.learning_rate = stud_lr
    config.batch_size = stud_bs

print(f"{TYPE_DATA}  {DISTRIBUTION}")

sys.stdout = open(f"log/{TYPE_DATA.lower()}_{DISTRIBUTION.lower()}_{compress_type}.txt", "w")
sys.stdout = Unbuffered(sys.stdout)

torch.manual_seed(0)

if DISTRIBUTION == 'SAMPLE':
    trainset = SelectedDataset(f'data/aided_sample/{TYPE_DATA}_selected', TYPE_DATA, one_hot=True, repeat=10)
    stud_bs=10
else:
    dataset = [f'pseudo_data/{DISTRIBUTION.lower()}_{TYPE_DATA.lower()}.json',]
    trainset = PseudoDataset(dataset)
train_loader = torch.utils.data.DataLoader(trainset, stud_bs)

_, devset = prepare_data("./data", TYPE_DATA)
dev_loader = torch.utils.data.DataLoader(devset, batch_size=128)

student = prepare_classifier(cls_type, im_size[0], 10)
state_dict = torch.load(f"ckpts/teacher/classifier_{TYPE_DATA.lower()}_{cls_type.lower()}.pt")
student.load_state_dict(state_dict["model_state"])

size_before = count_parameters(student)
student.compress(compress_type)

size_after = count_parameters(student)
print(f'size before:{size_before}, size after:{size_after} | {size_before/size_after} times')

optimizer = torch.optim.Adam(student.parameters(), lr = stud_lr)
criterion = KLDivLoss() 

trainer = Trainer(student, optimizer, criterion, train_loader, dev_loader, best_save_path="ckpts/student", use_wandb=use_wandb )
# trainer.load("ckpts_student/best.pt")

trainer.validate(0)
trainer.train(stud_epochs, print_freq=500)

sys.stdout.close()
