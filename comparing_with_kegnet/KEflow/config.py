# CLSTYPE in ["BASICCNN", "LENET5", "RESNET", "VGG"]
TYPE_CLS = "RESNET"
# TYPE in ["GLOW", "NICE", "REALNVP"]
TYPE_FLOW = "NICE"
# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "SVHN"

'''classifier config'''
CLS_CONFIG = {
    # train_config
    "BATCH_SIZE":32,
    "LR": 1e-4,
    "WD": 5e-4,
    "EPOCHS": 10,
    "PRINT_FREQ":500,
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
    "NUM_AIDED_SAMPLE":60000,
    # Extractor config
    "ALPHA":0.1,
    "BETA": 10, 
    # train config
    "LR":5e-5,
    "WD":1e-4,
    "SMOOTHE":0.,
    "BATCH_SIZE":64,
    "AIDED_BATCH_SIZE":64,
    "EPOCHS":50,
    "PRINT_FREQ":50,
    "VAL_FREQ":1,
}



