import copy
import os
from random import shuffle
from typing import *

from torch.utils.data import DataLoader
from tqdm import tqdm

from cusdataset import MyDataset
from utils import *


class FullBuffer:
    def __init__(self) -> None:
        self.buffer = MyDataset({})

    def append(self, dataset: MyDataset):
        self.buffer += dataset

    def create_loader(self, batch_size: int = 32, shuffle: bool = True):
        return DataLoader(self.buffer, batch_size=batch_size, shuffle=shuffle)


class FixedSizeBuffer:
    def __init__(self, max_size: int = 512, classes: bool | int = False) -> None:
        self.buffer = MyDataset({})
        self.max_size = max_size
        self.classes = classes

    def append(self, dataset: MyDataset):
        if self.classes == False:
            new_buffer = copy.deepcopy(dataset)
            indices = torch.randperm(len(self.buffer))[: self.max_size - len(dataset)]
            subset = self.buffer.subset(indices)
            new_buffer += subset
            self.buffer = new_buffer
        else:
            current_buffer = copy.deepcopy(dataset)

            # get_class idx in current buffer
            all_idx = list(range(len(self.buffer)))
            c_idx = {}
            for c in range(self.classes):
                c_idx[c] = []

            if len(self.buffer) != 0:
                for idx, d in enumerate(self.buffer):
                    c_idx[d["labels"].item()].append(idx)

            for c in range(self.classes):
                shuffle(c_idx[c])

            # calculate number of instances to be added
            counts = current_buffer.label_counts()
            proba = counts / counts.sum()

            need = (self.max_size / self.classes) * torch.ones(counts.shape) - counts
            available = torch.tensor([len(item) for item in c_idx.values()])
            added = torch.min(need, available)
            rest = available - added
            missing = self.max_size - (added.sum().item() + counts.sum().item())

            added_idx = []
            for idx, num in enumerate(need):
                indices = c_idx[idx][: num.int().item()]
                added_idx += indices
                current_buffer = current_buffer.__add__(self.buffer.subset(indices))

            pad = list(set(all_idx) - set(added_idx))[0 : int(missing)]
            if len(pad) > 0:
                current_buffer = current_buffer.__add__(self.buffer.subset(pad))
            self.buffer = current_buffer

    def create_loader(self, batch_size: int = 32, shuffle: bool = True):
        return DataLoader(self.buffer, batch_size=batch_size, shuffle=shuffle)


# # Paths
# CUR_DIR = os.path.abspath(os.curdir)
# CACHE_DIR = "D:/odl_cache/.cache/"
# SESSION_DIR = "D:/odl_cache/.cache/sessions_256/"
# CONFIG_DIR = os.path.join(CUR_DIR, "./configs/")
# LOG_DIR = "D:/model_zoo/odl/"

# # Settings
# NUM_SESSIONS = len(os.listdir(SESSION_DIR)) - 1


# def main():
#     for filename in os.listdir(CONFIG_DIR):
#         config_name = filename.split(".")[0]
#         log_dir = os.path.join(LOG_DIR, config_name)

#         # load configs
#         configs = read_config(os.path.join(CONFIG_DIR, filename))

#         # memory buffer
#         test_buffer = FullBuffer()  # to store all test data
#         if configs.get("buffer") == "full":  # for full rehearsal
#             buffer = FixedSizeBuffer(max_size=512, classes=2)
#         else:
#             buffer = None

#         for session in range(NUM_SESSIONS):
#             print(f"Session {session}")
#             # load session data from cache
#             cache = torch.load(os.path.join(SESSION_DIR, f"session_{session}.pt"))
#             train_dict = cache.get("dataset")["train"]
#             test_dict = cache.get("dataset")["test"]

#             # create dataset
#             train_dataset = MyDataset(train_dict)
#             test_dataset = MyDataset(test_dict)
#             test_dataset, eval_dataset = test_dataset.split(shuffle=False)

#             # create train_loader
#             if buffer:
#                 buffer.append(train_dataset)
           