import torch
from kegnet.classifier.train import main as train_student
from kegnet.generator.train import main as train_generator
from config import TYPE_DATA, cls_type

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if TYPE_DATA == "DIGIT":
    dataset = 'mnist'
if TYPE_DATA == 'FASHION':
    dataset = 'fashion'
if TYPE_DATA == 'SVHN':
    dataset = 'svhn'

path_teacher = f'ckpts/teacher/classifier_{TYPE_DATA.lower()}_{cls_type.lower()}.pt'
path_out = f'ckpts/generator/kegnet_{TYPE_DATA.lower()}'


path_gen = f'{path_out}'
path_model = train_generator(dataset, path_teacher, path_gen, 0)
