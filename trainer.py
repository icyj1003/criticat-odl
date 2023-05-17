from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metrics import accuracy


class Trainer:
    """Trainer
    Use for offline-batch training settings
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        optimizer: Callable,
        criterion: Callable,
        train_iter: DataLoader,
        eval_iter: DataLoader = None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        wandb_settings: dict = None,
    ) -> None:
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer(model.parameters())
        self.criterion = criterion
        self.device = device
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.wandb_settings = wandb_settings

    def train(self, epoch, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for texts, labels in tqdm(
            iterator, desc=f"Epoch: {epoch:03}", unit="steps", leave=False
        ):
            self.optimizer.zero_grad()

            texts = texts.to(self.device)
            labels = labels.to(self.device)
            predictions = self.model(texts)

            loss = self.criterion(predictions, labels)
            acc = accuracy(predictions, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() 
            epoch_acc += acc

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def eval(self, epoch, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for texts, labels in tqdm(
                iterator, desc=f"Epoch: {epoch:03}", unit="steps", leave=False
            ):
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(texts)

                loss = self.criterion(predictions, labels)
                acc = accuracy(predictions, labels)

                epoch_loss += loss.item()
                epoch_acc += acc

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train(epoch, self.train_iter)
            if self.eval_iter:
                eval_loss, eval_acc = self.eval(epoch, self.eval_iter)

                # logs to stdout
                if not self.wandb_settings:
                    print(
                        f"Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Eval Loss: {eval_loss:.3f} | Eval Acc: {eval_acc:.2f}%"
                    )

                # use wandb
                else:
                    pass
            # logs to stdout
            if not self.wandb_settings:
                print(
                    f"Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%"
                )
            # use wandb
            else:
                pass
