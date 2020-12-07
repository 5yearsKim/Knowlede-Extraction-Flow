# CLSTYPE in ["BASICCNN", "LENET5", "RESNET", "VGG", "WRN"]
TYPE_CLS = "BASICCNN"
# TYPE in ["GLOW", "NICE", "REALNVP"]
TYPE_FLOW = "NICE"
# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "SVHN"

'''classifier config'''
CLS_CONFIG = {
    # train_config
    "BATCH_SIZE":128,
    "LR": 1e-4,
    "WD": 5e-4,
    "EPOCHS": 5,
    "PRINT_FREQ":1000,
    "VAL_FREQ": 1,
    # model config
    "NC":3,
    "N_CLASS":10,
    "IM_SIZE":32,
}

""" NICE Extractor config """
FLOW_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 10000,
    # Extractor config
    "SPREAD_S":0.05,
    "GRAVITY_S": 0., 
    "BN_S": 10.,
    # train config
    "LR":5e-5,
    "WD":1e-4,
    "SMOOTHE":0.,
    "BATCH_SIZE":64,
    "EPOCHS":100,
    "PRINT_FREQ":500,
    "VAL_FREQ":1,
}



