
'''classifier config'''
CLS_CONFIG = {
    # train_config
    "BATCH_SIZE":32,
    "LR": 1e-3,
    "WD": 1e-2,
    "EPOCHS": 2,
    "PRINT_FREQ":250,
    "VAL_FREQ": 1,
    # model config
    "NC":1,
    "IM_SIZE":32,
    "N_FILTER":32,
    "N_CLASS":10,
}

FLOW_CONFIG = {
    # Dataset
    "NUM_SAMPLE" : 2048,
    "NUM_AIDED_SAMPLE":2048,
    # model config
    "COUPLING": 16,
    "IN_OUT_DIM": 32*32,
    "COND_DIM": 10,
    "MID_DIM": 1500,
    "HIDDEN":2, 
    # train config
    "LR":3e-4,
    "WD":2e-3,
    "SMOOTHE":0.1,
    "BATCH_SIZE":32,
    "AIDED_BATCH_SIZE":16,
    "EPOCHS":10,
    "PRINT_FREQ":100,
    "VAL_FREQ":1
}
