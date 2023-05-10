import json

import torch
from tqdm.auto import tqdm

from preprocess import VietnameseTextCleaner


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
            result[k] = result.get(k, []) + [v]
    return result


def DLtoLD(d):
    if not d:
        return []
    result = [{} for i in range(max(map(len, d.values())))]
    for k, seq in d.items():
        for oneDict, oneValue in zip(result, seq):
            oneDict[k] = oneValue
    return result


def generate_toy_dataset(length: int = 100):
    pass


def read_config(filename):
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
    """AI is creating summary for eval_one

    Args:
        model ([type]): [description]
        criterion ([type]): [description]
        inputs ([type]): [description]
        targets ([type]): [description]

    Returns:
        [type]: [description]
    """
    # set the PyTorch model to evaluating mode
    model.eval()

    with torch.no_grad():
        # forward pass
        outputs = model(*inputs)

    # calculate the loss
    loss = criterion(outputs, targets)

    return loss, outputs


def sum(a, b):
    """AI is creating summary for sum

    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    return a + b
