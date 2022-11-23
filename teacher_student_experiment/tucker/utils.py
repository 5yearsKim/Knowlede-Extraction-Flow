import numpy as np

def count_parameters(model):
    """
    Count the parameters of a classifier.
    """
    size = 0
    for parameter in model.parameters():
        size += np.prod(parameter.shape)
    return size