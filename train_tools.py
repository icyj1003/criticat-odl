import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from callbacks import EarlyStopping


def train_offline(dataloader, F, G, loss_function, optimizer, device):
    epoch_loss = 0

    y_true = []
    y_pred = []
    proba = []

    G.train()
    F.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        inputs = batch["post_message"]
        labels = batch["label"].to(device)
        embedded = G(inputs)
        output = F(embedded)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / len(dataloader)
        y_true.extend(labels.tolist())
        y_pred.extend(output.argmax(dim=1).tolist())
        proba.extend(output[:, 1].tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return epoch_loss, acc, f1, precision, recall, proba


def eval_offline(dataloader, F, G, loss_function, device):
    epoch_loss = 0

    y_true = []
    y_pred = []
    proba = []

    G.eval()
    F.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch["post_message"]
            labels = batch["label"].to(device)
            embedded = G(inputs)
            output = F(embedded)
            loss = loss_function(output, labels)

            epoch_loss += loss.item() / len(dataloader)
            y_true.extend(labels.tolist())
            y_pred.extend(output.argmax(dim=1).tolist())
            proba.extend(output[:, 1].tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return epoch_loss, acc, f1, precision, recall, proba


# def train_epoch(
#     epoch: int,
#     G: nn.Module,
#     F: nn.Module,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: nn.Module,
#     device: torch.device,
# ):
#     y_true = []
#     y_pred = []
#     epoch_loss = 0

#     G.train()
#     F.train()
#     for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
#         optimizer.zero_grad()
#         inputs = batch["post_message"]
#         labels = batch["label"].to(device)
#         embedded = G(inputs)
#         output = F(embedded)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()

#         y_true.extend(labels.tolist())
#         y_pred.extend(output.argmax(dim=1).tolist())

#         epoch_loss += loss.item() / len(loader)

#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average="macro")

#     return epoch_loss, acc, f1


# def eval_epoch(
#     epoch: int,
#     G: nn.Module,
#     F: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     device: torch.device,
# ):
#     y_true = []
#     y_pred = []
#     epoch_loss = 0

#     G.eval()
#     F.eval()
#     with torch.no_grad():
#         for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
#             inputs = batch["post_message"]
#             labels = batch["label"].to(device)
#             embedded = G(inputs)
#             output = F(embedded)
#             loss = criterion(output, labels)

#             y_true.extend(labels.tolist())
#             y_pred.extend(output.argmax(dim=1).tolist())

#             epoch_loss += loss.item() / len(loader)

#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average="macro")

#     return epoch_loss, acc, f1


# def train_offline(
#     train_dataset,
#     test_dataset,
#     batch_size,
#     G: nn.Module,
#     F: nn.Module,
#     criterion,
#     optimizer,
#     num_epochs,
#     device,
#     early_stopping=True,
#     patience=5,
#     verbose=True,
#     delta=0,
#     path="./",
# ):
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     if early_stopping:
#         es = EarlyStopping(
#             patience=patience,
#             verbose=verbose,
#             delta=delta,
#             path=path,
#         )

#     for epoch in range(num_epochs):
#         train_loss, train_acc, train_f1 = train_epoch(
#             epoch, G, F, train_loader, optimizer, criterion, device
#         )
#         test_loss, test_acc, test_f1 = eval_epoch(
#             epoch, G, F, test_loader, criterion, device
#         )
#         if verbose:
#             print(
#                 f"Epoch {epoch} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - train_f1: {train_f1:.4f} - test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f} - test_f1: {test_f1:.4f}"
#             )

#         if early_stopping:
#             es(test_loss, test_f1, test_acc, G, F)
#             if es.early_stop:
#                 if verbose:
#                     print("Early stopping")
#                 break
