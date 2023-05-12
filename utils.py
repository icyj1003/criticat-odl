import datetime
import json
from typing import *

import torch
from tqdm.auto import tqdm

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


def LDtoDL(l):
    result = {}
    for d in l:
        for k, v in d.items():
            t = result.get(k)
            if t is not None:
                result[k] = torch.cat([t, v])
            else:
                result[k] = v
    return result


def DLtoLD(d):
    if not d:
        return []
    result = [{} for i in range(max(map(len, d.values())))]
    for k, seq in d.items():
        for oneDict, oneValue in zip(result, seq):
            oneDict[k] = oneValue
    return result


def read_config(filename) -> dict:
    with open(filename, "r") as file:
        data = json.load(file)

    return data


def train_one(model, optimizer, criterion, inputs, targets):
    # set the PyTorch model to training mode
    model.train()

    # zero-out the gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(*inputs)

    # calculate the loss
    loss = criterion(outputs, targets)

    # backward propagation
    loss.backward()

    # optimization step
    optimizer.step()

    return loss, outputs


def eval_one(model, criterion, inputs, targets):
    # set the PyTorch model to evaluating mode
    model.eval()

    with torch.no_grad():
        # forward pass
        outputs = model(*inputs)

    # calculate the loss
    loss = criterion(outputs, targets)

    return loss, outputs


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
