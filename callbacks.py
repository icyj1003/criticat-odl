import os
from typing import *

import numpy as np
import torch


# Early stoping class in pytorch and save best model (with 2 nn.Module G and F) base on f1 score
class EarlyStopping:
    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="./",
        model_G=None,
        model_F=None,
    ):
        """
        Early stopping to stop the training when the loss does not improve after
        certain epochs and save the best model base on f1 score
        :param patience: How long to wait after last time validation loss improved.
        :param verbose: If True, prints a message for each validation loss improvement.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        :param path: Path for the checkpoint to be saved to.
        :param model_G: model G
        :param model_F: model F
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.f1_score = -np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_min = np.Inf
        self.delta = delta
        self.path = path
        self.model_G = model_G
        self.model_F = model_F

    def __call__(self, val_loss, f1_score, val_acc, model_G, model_F):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, f1_score, val_acc, model_G, model_F)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience} with f1 score {score}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, f1_score, val_acc, model_G, model_F)
            self.counter = 0

    def save_checkpoint(self, val_loss, f1_score, val_acc, model_G, model_F):
        """
        Save the best model base on f1 score
        :param val_loss: validation loss
        :param f1_score: f1 score
        :param model_G: model G
        :param model_F: model F
        :return:
        """

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}), f1 score increased ({self.f1_score:.6f} --> {f1_score:.6f}).  Saving model ..."
            )
        torch.save(
            {"model_G": model_G.state_dict(), "model_F": model_F.state_dict()},
            os.path.join(self.path, "checkpoint.pt"),
        )
        self.val_loss_min = val_loss
        self.f1_score = f1_score
        self.val_acc_min = val_acc
