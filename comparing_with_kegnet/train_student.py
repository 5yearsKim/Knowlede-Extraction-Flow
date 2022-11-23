from kegnet.classifier.train import main as train_student

dataset = 'svhn'
data_dist = 'flow'
option = 3
seed = 0
path_teacher = f'pretrained/svhn.pth.tar'
generators = [f'ckpts/extractor3.pt' ,f'ckpts/extractor4.pt']
path_out = 'ckpt_flow_student'

train_student(dataset, data_dist, path_out, seed, path_teacher, generators,
                option)
