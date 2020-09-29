
'''classifier config'''
CLS_CONFIG = {
    # Dataset
    "NUM_SAMPLE": 1024,
    "TOY_TYPE": "circle",
    # train_config
    "BATCH_SIZE":16,
    "LR": 1e-3,
    "EPOCHS": 40,
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
    "NUM_SAMPLE": 2048,
    # model config
    "COUPLING": 6,
    "IN_OUT_DIM": 2,
    "COND_DIM": 2,
    "MID_DIM": 40,
    "HIDDEN":3, 
    # train config
    "LR":1e-3,
    "BATCH_SIZE":32,
    "EPOCHS":20,
    "PRINT_FREQ":20,
    "VAL_FREQ":1
}
