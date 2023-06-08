import argparse
import json
import os
import random
from typing import *

import numpy as np
import torch
from sklearn.metrics import *


def create_namespace(args_dict):
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    return args


def load_config_from_file(file_path) -> dict:
    jsondata = ""
    with open(file_path, "r") as jsonfile:
        for line in jsonfile:
            jsondata += line.split("//")[0]

    objec = json.loads(jsondata)

    return objec


def save_json(dir, filename, object):
    with open(os.path.join(dir, filename), "w") as fp:
        json.dump(object, fp)

def load_json(dir, filename):
    with open(os.path.join(dir, filename), "r") as fp:
        return json.load(fp)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
