import datetime
import json
import os
from pathlib import Path
from typing import *

import torch
from sklearn.metrics import accuracy_score

from metrics import MeanMetric
from module import *
from preprocess import VietnameseTextCleaner
from transforms import TextTransform
from vocabulary import Vocabulary


def to_number(object):
    num = 0
    try:
        num = int(object)
    except:
        pass
    return num


def dict_handler(
    cleaner: VietnameseTextCleaner,
    dict_object: dict = None,
    islabeled=True,
):
    # create a copy of object
    data = dict_object

    # preprocess
    data["num_share_post"] = to_number(data["num_share_post"])
    data["num_like_post"] = to_number(data["num_like_post"])
    data["num_comment_post"] = to_number(data["num_comment_post"])
    data["raw_length"] = len(str(data["post_message"]))
    data["post_message"] = cleaner.clean_one(data["post_message"])
    return data


def read_config(filename) -> dict:
    jsondata = ""
    with open(filename, "r") as jsonfile:
        for line in jsonfile:
            jsondata += line.split("//")[0]

    objec = json.loads(jsondata)

    return objec


def repare_input(configs, batch, device):
    if configs["type"] == "ensemble" or configs["type"] == "text":
        vectors = batch["texts"].to(device)

    if configs["type"] == "ensemble" or configs["type"] == "image":
        images = batch["images"].to(device, dtype=torch.float)

    if configs["type"] == "ensemble" or configs["type"] == "metadata":
        metadata = batch["metadata"].to(device)

    if configs["type"] == "ensemble" or configs["type"] == "user_name":
        user_name = batch["user_name"].to(device)
    labels = batch["labels"].to(device)

    if configs["type"] == "ensemble":
        return (vectors, images, user_name, metadata), labels
    elif configs["type"] == "text":
        return (vectors,), labels
    elif configs["type"] == "image":
        return (images,), labels
    elif configs["type"] == "metadata":
        return (metadata,), labels
    elif configs["type"] == "user_name":
        return (user_name,), labels


def train_epoch(model, optimizer, criterion, loader, device, configs):
    # default values
    y_true = []
    y_pred = []
    epoch_loss = MeanMetric()

    # training mode
    model.train()
    for batch in loader:
        inputs, labels = repare_input(configs, batch, device)

        # zero-out the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(*inputs)

        # calculate the loss
        loss = criterion(outputs, labels)

        # backward propagation
        loss.backward()

        # optimization step
        optimizer.step()

        y_true.append(labels.cpu())
        y_pred.append(torch.argmax(outputs, dim=1).cpu())

        epoch_loss.update(loss.item())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    acc = accuracy_score(y_true, y_pred)

    return epoch_loss.result(), acc, y_true, y_pred


def eval_epoch(model, criterion, loader, device, configs):
    # default values
    y_true = []
    y_pred = []
    epoch_loss = MeanMetric()

    # eval mode
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs, labels = repare_input(configs, batch, device)

            # forward pass
            outputs = model(*inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            y_true.append(labels.cpu())
            y_pred.append(torch.argmax(outputs, dim=1).cpu())

            epoch_loss.update(loss.item())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    acc = accuracy_score(y_true, y_pred)

    return epoch_loss.result(), acc, y_true, y_pred


def handling_text(
    texts: Iterable, vocab: Vocabulary, train: bool = True
) -> torch.Tensor:
    tokenizer = lambda x: [_.split() for _ in x]
    text_transform = TextTransform(max_length=20)
    texts = tokenizer(texts)
    if train:
        # update token to vocab
        vocab.append_from_iterator(texts)
    # token -> index -> padding -> tensor -> batch
    texts = torch.stack([text_transform(vocab.get_idxs(tokens)) for tokens in texts])
    return texts


def handling_username(
    usernames: Iterable, LUT: Vocabulary, train: bool = True
) -> torch.Tensor:
    if train:
        LUT.append_tokens(usernames)
    # username -> index
    return torch.stack([torch.tensor(LUT[user]) for user in usernames])


def handling_metadata(
    num_like_post: Iterable,
    num_comment_post: Iterable,
    num_share_post: Iterable,
    raw_length: Iterable,
    timestamp_post: Iterable,
):
    hours = torch.tensor(
        [datetime.datetime.fromtimestamp(x.item()).hour for x in timestamp_post],
        dtype=torch.float,
    )
    weekdays = torch.tensor(
        [datetime.datetime.fromtimestamp(x.item()).weekday() for x in timestamp_post],
        dtype=torch.float,
    )

    return (
        torch.stack(
            [
                num_like_post,
                num_comment_post,
                num_share_post,
                raw_length,
                weekdays,
                hours,
            ],
            dim=1,
        )
        + 1
    ).log()


def load_model_from_config(configs):
    # text model
    if configs["type"] == "ensemble" or configs["type"] == "text":
        if configs["text_model"]["architecture"] == "lstm":
            text_model = LSTMClassifier(
                embedding_dim=configs["text_model"].get("embedding_dim"),
                hidden_dim=configs["text_model"].get("hidden_dim"),
                vocab_size=configs["text_model"].get("max_features"),
                dropout_rate=configs["text_model"].get("dropout_rate"),
                use_pretrain_embedding=configs["text_model"].get(
                    "use_pretrain_embedding"
                ),
            )
        elif configs["text_model"]["architecture"] == "gru":
            text_model = GRUClassifier(
                embedding_dim=configs["text_model"].get("embedding_dim"),
                hidden_dim=configs["text_model"].get("hidden_dim"),
                vocab_size=configs["text_model"].get("max_features"),
                dropout_rate=configs["text_model"].get("dropout_rate"),
                use_pretrain_embedding=configs["text_model"].get(
                    "use_pretrain_embedding"
                ),
            )
        elif configs["text_model"]["architecture"] == "bilstm":
            text_model = LSTMClassifier(
                embedding_dim=configs["text_model"].get("embedding_dim"),
                hidden_dim=configs["text_model"].get("hidden_dim"),
                vocab_size=configs["text_model"].get("max_features"),
                dropout_rate=configs["text_model"].get("dropout_rate"),
                bidirectional=True,
                use_pretrain_embedding=configs["text_model"].get(
                    "use_pretrain_embedding"
                ),
            )
        elif configs["text_model"]["architecture"] == "bigru":
            text_model = GRUClassifier(
                embedding_dim=configs["text_model"].get("embedding_dim"),
                hidden_dim=configs["text_model"].get("hidden_dim"),
                vocab_size=configs["text_model"].get("max_features"),
                dropout_rate=configs["text_model"].get("dropout_rate"),
                bidirectional=True,
                use_pretrain_embedding=configs["text_model"].get(
                    "use_pretrain_embedding"
                ),
            )
        else:
            raise Exception("Unknown text model architecture!")
    if configs["type"] == "ensemble" or configs["type"] == "image":
        if configs["image_model"]["architecture"] == "resnet18":
            image_model = Resnet18()
        elif configs["image_model"]["architecture"] == "resnet34":
            image_model = Resnet34()
        else:
            raise Exception("Unknown image model architecture!")

    if configs["type"] == "ensemble" or configs["type"] == "metadata":
        metadata_model = Metadata(
            input_size=6, hidden_size=configs["metadata_model"].get("hidden_size")
        )
    if configs["type"] == "ensemble" or configs["type"] == "user_name":
        user_model = UserEmbedding(
            max_num_user=configs["user_model"].get("max_user"),
            embedding_dim=configs["user_model"].get("embedding_dim"),
            dropout_rate=configs["user_model"].get("dropout_rate"),
        )
    if configs["type"] == "ensemble":
        model = Ensemble(
            text_model=text_model,
            image_model=image_model,
            user_model=user_model,
            metadata_model=metadata_model,
        )
        return model
    elif configs["type"] == "text":
        return text_model
    elif configs["type"] == "image":
        return image_model
    elif configs["type"] == "metadata":
        return metadata_model
    elif configs["type"] == "user_name":
        return user_model


def save_object(object, dir, filename=None, mode: Literal["pt", "jsonc"] = "pt"):
    # ensure dir exists
    Path(dir).mkdir(parents=True, exist_ok=True)

    # save object
    if mode == "pt":
        torch.save(object, os.path.join(dir, f"{filename}.pt"))
    elif mode == "jsonc":
        assert type(object) is dict, "type must be dict when saving as jsonc"
        with open(os.path.join(dir, f"{filename}.jsonc"), "w") as outfile:
            json.dump(object, outfile)
