from kegnet.classifier.train import main as train_teacher

dataset = "mnist"
data_dist = "real"
path_out = "out"

train_teacher(dataset, data_dist, path_out)