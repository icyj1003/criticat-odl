import os
from typing import *

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from buffer import Buffer, SimpleBuffer
from callbacks import EarlyStopping


def train_epoch(
    dataloader, input_fields, F, G, optimizer, loss_function, device
):
    if G is not None:
        G.train()
    F.train()

    epoch_loss = 0
    y_true = []
    y_pred = []
    proba = []

    for batch in tqdm(dataloader, leave=False):
        optimizer.zero_grad()

        inputs = (batch[field] for field in input_fields)
        labels = batch["label"].to(device)

        encoded_inputs = G(*inputs)

        if len(input_fields) == 1:
            encoded_inputs = torch.unsqueeze(encoded_inputs, 0)

        outputs = F(*encoded_inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).tolist())
        proba.extend(outputs[:, 1].tolist())
        epoch_loss += loss.item() / len(dataloader)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return epoch_loss, acc, f1, precision, recall, proba


def eval_epoch(dataloader, input_fields, F, G, loss_function, device):
    if G is not None:
        G.eval()
    F.eval()

    epoch_loss = 0
    y_true = []
    y_pred = []
    proba = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            inputs = (batch[field] for field in input_fields)
            labels = batch["label"].to(device)

            encoded_inputs = G(*inputs)

            if len(input_fields) == 1:
                encoded_inputs = torch.unsqueeze(encoded_inputs, 0)

            outputs = F(*encoded_inputs)

            loss = loss_function(outputs, labels)

            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).tolist())
            proba.extend(outputs[:, 1].tolist())
            epoch_loss += loss.item() / len(dataloader)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return epoch_loss, acc, f1, precision, recall, proba


def eval(test_dataset, input_fields, F, G, loss_function, batch_size, device):
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return eval_epoch(test_loader, input_fields, F, G, loss_function, device)


def extract_features(dataset, input_fields, G, batch_size, device):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    G.eval()

    features = []
    targets = []

    for batch in tqdm(loader, leave=False):
        inputs = (batch[field] for field in input_fields)
        labels = batch["label"]

        # pass input through G
        with torch.no_grad():
            encoded_inputs = G(*inputs)

        # handling input shape for multimodal
        if len(input_fields) == 1:
            # create a sequence of input fields
            # equivalent to [encoded_inputs]
            encoded_inputs = [
                encoded_inputs
            ]  # torch.unsqueeze(encoded_inputs, 0)

        for _ in zip(*encoded_inputs):
            features.append([a.cpu() for a in _])
        targets.append(labels)
    return features, torch.cat(targets)


def train_offline(
    train_dataset,
    dev_dataset,
    input_fields,
    G,
    F,
    optimizer,
    loss_function,
    batch_size,
    device,
    early_stopping,
    checkpoint_path,
    epochs=100,
    patience=5,
    delta=0.001,
    verbose=True,
):
    es = EarlyStopping(
        patience=patience, delta=delta, verbose=verbose, path=checkpoint_path
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False
    )

    for epoch in range(1, 1 + epochs):
        train_loss, train_acc, train_f1, _, _, _ = train_epoch(
            dataloader=train_loader,
            input_fields=input_fields,
            F=F,
            G=G,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        dev_loss, dev_acc, dev_f1, _, _, _ = eval_epoch(
            dataloader=dev_loader,
            input_fields=input_fields,
            F=F,
            G=G,
            loss_function=loss_function,
            device=device,
        )

        if verbose:
            print(
                f"epoch: {epoch:02} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | dev_loss {dev_loss:.4f} | dev_acc: {dev_acc:.4f} | dev_f1: {dev_f1:.4f}"
            )

        es(dev_loss, G, F, optimizer)

        if es.early_stop and early_stopping:
            if verbose:
                print("Early stop, restore best weights")
            torch.save(
                {
                    "model_G": G.state_dict(),
                    "model_F": F.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(checkpoint_path, "checkpoint_last.pt"),
            )
            break
        del dev_loss, dev_acc, dev_f1, train_loss, train_acc, train_f1, _
    es.restore_best_weights(G, F, optimizer)
    del es, train_loader, dev_loader


def train_online(
    online_dataset,
    input_fields,
    G,
    F,
    optimizer,
    loss_function,
    device,
    buffer_size=64,
    buffer: Union[None, Buffer] = None,
):
    online_loader = torch.utils.data.DataLoader(
        online_dataset, batch_size=1, shuffle=False
    )

    G.eval()
    F.train()

    for batch in tqdm(online_loader, leave=False):
        optimizer.zero_grad()

        inputs = (batch[field] for field in input_fields)
        labels = batch["label"]

        # pass input through G
        with torch.no_grad():
            encoded_inputs = G(*inputs)

        # handling input shape for multimodal
        if len(input_fields) == 1:
            # create a sequence of input fields
            # equivalent to [encoded_inputs]
            encoded_inputs = [encoded_inputs]

        # buffer mix here:
        if buffer is not None:
            encoded_inputs, labels = buffer.mix(
                encoded_inputs, labels, old_samples=buffer_size, device=device
            )
        else:
            labels = labels.to(device)

        outputs = F(*encoded_inputs)

        del encoded_inputs

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
