import copy
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

        # setting
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        # monitoring
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

        # save dir
        self.path = path

        # best states
        self.optimizer = None
        self.model_G = model_G
        self.model_F = model_F

    def __call__(self, val_loss, model_G, model_F, optimizer):
        loss = val_loss

        # first call or loss decrease detected
        if (self.val_loss_min == np.Inf) or (
            loss + self.delta < self.val_loss_min
        ):
            if self.verbose:
                print(
                    f"validation loss decreased ({self.val_loss_min:.6f} --> {loss:.6f}).  Saving model ..."
                )
            # save current state
            self.save_checkpoint(model_G, model_F, optimizer)
            self.val_loss_min = loss
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience} with val loss {loss}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model_G, model_F, optimizer):
        """
        Save the best model base on f1 score
        :param model_G: model G
        :param model_F: model F
        :param optimizer: optimizer
        :return:
        """
        torch.save(
            {
                "G": model_G.state_dict(),
                "F": model_F.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(self.path, "checkpoint_best.pt"),
        )
        self.model_F = copy.deepcopy(model_F.state_dict())
        self.model_G = copy.deepcopy(model_G.state_dict())
        self.optimizer = copy.deepcopy(optimizer.state_dict())

    def restore_best_weights(self, model_G, model_F, optimizer):
        model_G.load_state_dict(self.model_G)
        model_F.load_state_dict(self.model_F)
        optimizer.load_state_dict(self.optimizer)
