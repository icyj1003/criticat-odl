import torch.nn.functional as f
import torch
from utils import read_config
import os
import wandb
import time

# Paths
CUR_DIR = os.path.abspath(os.curdir)
CACHE_DIR = os.path.join(CUR_DIR, "./.cache/")
CONFIG_DIR = os.path.join(CUR_DIR, "./configs/")

# load config file
configs = read_config("./configs/train.jsonc")

# Init wandb
wandb.init(project="odl", config=configs)

for i in range(100):
    acc = torch.randint(0, 100, (1,))
    wandb.log({"acc": acc})
    time.sleep(1)
