# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "DIGIT"
# DISTRIBUTION in ["UNIFORM", "NORMAL", "KEGNET", "KEFLOW", "KEFLOW+", "SAMPLE"]
DISTRIBUTION = "KEFLOW+"

compress_type =  1 
# Common param
NUM_BATCH = 100
BATCH_SIZE = 64

flow_bs = 64
flow_epochs = 30
flow_print_freq = 150

stud_bs = 128
stud_epochs = 200

# Data specific param
if TYPE_DATA == "DIGIT":
    im_size = (1, 32, 32)
    cls_type = "LENET5"
    flow_lr = 5e-5
    stud_lr = 1e-4
    det_s = 5e-3
    bn_s = 0
    aided_weight = 1e-3

if TYPE_DATA == "FASHION":
    im_size = (1, 32, 32)
    cls_type = "RESNET"
    flow_lr = 1e-4
    stud_lr = 1e-4
    det_s = 0.02
    bn_s = 2.
    aided_weight = .01

if TYPE_DATA == "SVHN":
    im_size = (3, 32, 32)
    cls_type = "RESNET"

    flow_lr = 1e-4
    stud_lr = 1e-4
    det_s = 0.04
    bn_s = 2.
    aided_weight = 1e-3

if TYPE_DATA == "CIFAR":
    im_size = (3, 32, 32)
    cls_type = "RESNET"
    flow_lr = 1e-4
    stud_lr = 1e-4
    det_s = 0.04
    bn_s = 0.5
    aided_weight = 1

#  if DISTRIBUTION in ['UNIFORM', 'NORMAL']:
#      stud_lr = 1e-7
