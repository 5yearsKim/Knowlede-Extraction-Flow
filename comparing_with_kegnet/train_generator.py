from kegnet.generator.train import main as train_generator

dataset = "mnist"
path_teacher = "ckpt_teacher/classifier-best.pth.tar"
path_gen = "ckpt_generator/"

path_model = train_generator(dataset, path_teacher, path_gen, 0)