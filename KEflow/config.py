# CLSTYPE in ["BASICCNN", "LENET5", "RESNET", "VGG"]
TYPE_CLS = "BASICCNN"
# TYPE in ["GLOW", "NICE", "REALNVP"]
TYPE_FLOW = "REALNVP"
# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "FASHION"

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
    "NUM_SAMPLE" : 1000,
    "NUM_AIDED_SAMPLE":1024,
    # Extractor config
    "ALPHA":0.01,
    "BETA": 1, 
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



