# TYPE in ["GLOW", "NICE"]
TYPE = "GLOW"

'''classifier config'''
CLS_CONFIG = {
    # dataset
    "TYPE": "DIGIT",
    # train_config
    "BATCH_SIZE":32,
    "LR": 1e-4,
    "WD": 1e-5,
    "EPOCHS": 2,
    "PRINT_FREQ":500,
    "VAL_FREQ": 1,
    # model config
    "NC":1,
    "IM_SIZE":32,
    "N_FILTER":32,
    "N_CLASS":10,
}

""" NICE Extractor config """
NICE_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 10240,
    "NUM_AIDED_SAMPLE":20,
    # model config
    "COUPLING": 12,
    "COND_DIM": 10,
    "MID_DIM": 1024,
    "HIDDEN":2, 
    # train config
    "LR":3e-5,
    "WD":1e-4,
    "SMOOTHE":0.01   ,
    "BATCH_SIZE":32,
    "AIDED_BATCH_SIZE":8,
    "EPOCHS":5,
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
    # train config
    "LR":5e-5,
    "WD":1e-4,
    "SMOOTHE":0.01   ,
    "BATCH_SIZE":32,
    "AIDED_BATCH_SIZE":8,
    "EPOCHS":15,
    "PRINT_FREQ":10,
    "VAL_FREQ":1
}