import copy
from typing import *

import torch.nn as nn


class EarlyStopping:
    def __init__(
        self, tolerance=5, min_delta=0, mode: str = Literal["max", "min", "auto"]
    ):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        self.min_loss = float("inf")
        self.max_acc = 0

    def __call__(self, model: nn.Module, epoch, val_loss, val_acc):
        if self.mode == "min" or self.mode == "auto":
            if (self.min_loss - val_loss) >= self.min_delta:
                self.min_loss = val_loss
                self.weights = copy.deepcopy(model.state_dict())
                self.epoch = epoch
            else:
                self.counter += 1
                if self.counter >= self.tolerance:
                    self.early_stop = True
        if self.mode == "max" or self.mode == "auto":
            if (val_acc - self.max_acc) >= self.min_delta:
                self.max_acc = val_acc
                self.weights = copy.deepcopy(model.state_dict())
                self.epoch = epoch
            else:
                self.counter += 1
                if self.counter >= self.tolerance:
                    self.early_stop = True
