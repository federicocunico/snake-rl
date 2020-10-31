import numpy as np
import torch

from src.snake_structures import GridPosition


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    if isinstance(y, np.ndarray):
        return np.eye(num_classes, dtype='uint8')[y]

    return torch.eye(num_classes)[y]

