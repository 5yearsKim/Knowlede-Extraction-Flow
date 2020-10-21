# CLSTYPE in ["BASICCNN", "LENET5", "RESNET", "VGG"]
TYPE_CLS = "LENET5"
# TYPE in ["GLOW", "NICE", "REALNVP"]
TYPE_FLOW = "NICE"
# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "DIGIT"

'''classifier config'''
CLS_CONFIG = {
    # train_config
    "BATCH_SIZE":1,
    "LR": 1e-6,
    "WD": 5e-4,
    "EPOCHS": 10,
    "PRINT_FREQ":500,
    "VAL_FREQ": 1,
    # model config
    "NC":1,
    "N_CLASS":10,
    "IM_SIZE":32,
}

""" NICE Extractor config """
FLOW_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 10000,
    "NUM_AIDED_SAMPLE":10240,
    # Extractor config
    "ALPHA":0.02,
    "BETA": 2, 
    # train config
    "LR":5e-5,
    "WD":1e-4,
    "SMOOTHE":0.01,
    "BATCH_SIZE":32,
    "AIDED_BATCH_SIZE":32,
    "EPOCHS":50,
    "PRINT_FREQ":100,
    "VAL_FREQ":1
}



