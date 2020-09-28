
'''classifier config'''
CLS_CONFIG = {
    # Dataset
    "NUM_SAMPLE": 400,
    # train_config
    "BATCH_SIZE":8,
    "LR": 1e-3,
    "EPOCHS": 30,
    "PRINT_FREQ": 20,
    "VAL_FREQ": 1,
    # model config
    "DIM_IN":2,
    "DIM_OUT":2,
    "N_HIDDEN":2,
    "DIM_HIDDEN":32,
}

