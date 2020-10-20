# CLSTYPE in ["BASICCNN", "LENET5", "RESNET"]
TYPE_CLS = "BASICCNN"
# TYPE in ["GLOW", "NICE"]
TYPE_FLOW = "NICE"
# TYPE in ["DIGIT", "FASHION", "SVHN", "CIFAR"]
TYPE_DATA = "SVHN"

'''classifier config'''
CLS_CONFIG = {
    # train_config
    "BATCH_SIZE":64,
    "LR": 2e-4,
    "WD": 1e-3,
    "EPOCHS": 10,
    "PRINT_FREQ":500,
    "VAL_FREQ": 1,
    # model config
    "NC":3,
    "N_CLASS":10,
    "IM_SIZE":32,
}

""" NICE Extractor config """
NICE_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 10240,
    "NUM_AIDED_SAMPLE":1024,
    # model config
    "COUPLING": 12,
    "COND_DIM": 10,
    "MID_DIM": 1500,
    "HIDDEN":2,
    # Extractor config
    "ALPHA":0.01,
    "BETA": 1, 
    # train config
    "LR":5e-5,
    "WD":1e-4,
    "SMOOTHE":0.01,
    "BATCH_SIZE":24,
    "AIDED_BATCH_SIZE":16,
    "EPOCHS":50,
    "PRINT_FREQ":100,
    "VAL_FREQ":1
}

""" GLOW Extractor config """
GLOW_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 10240,
    "NUM_AIDED_SAMPLE":20,
    # model config
    "IN_CHANNELS":1,
    "MID_CHANNELS":32,
    "COND_DIM":10,
    "NUM_LEVELS":3,
    "NUM_STEPS":8,
    # extractor config
    "ALPHA" :0.01,   
    "BETA" :1,
    # train config
    "LR":1e-5,
    "WD":1e-4,
    "SMOOTHE":0.01   ,
    "BATCH_SIZE":64,
    "AIDED_BATCH_SIZE":8,
    "EPOCHS":10,
    "PRINT_FREQ":50,
    "VAL_FREQ":1
}
