
'''classifier config'''
CLS_CONFIG = {
    # Dataset
    "NUM_SAMPLE": 10240,
    "TOY_TYPE": "pizza",
    # train_config
    "BATCH_SIZE":32,
    "LR": 1e-3,
    "WD": 5e-2,
    "EPOCHS": 10,
    "PRINT_FREQ": 20,
    "VAL_FREQ": 1,
    # model config
    "DIM_IN":2,
    "DIM_OUT":2,
    "N_HIDDEN":2,
    "DIM_HIDDEN":32,
}

FLOW_CONFIG = {
    # Dataset
    "NUM_SAMPLE": 20000,
    "NUM_AIDED_SAMPLE":30,
    # model config
    "COUPLING": 6,
    "IN_OUT_DIM": 2,
    "COND_DIM": 2,
    "MID_DIM": 30,
    "HIDDEN":2, 
    # train config
    "LR":1e-3,
    "WD":5e-3,
    "SMOOTHE":0.05,
    "BATCH_SIZE":32,
    "AIDED_BATCH_SIZE":16,
    "EPOCHS":4,
    "PRINT_FREQ":100,
    "VAL_FREQ":1
}
