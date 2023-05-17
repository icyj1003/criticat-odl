from __future__ import annotations

import bisect
import json
import time
import warnings
from collections import Counter
from typing import *

import torch
from kafka import KafkaConsumer
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm


class TestingDataset(Dataset):
    def __init__(self, data: List[dict]) -> None:
        self.data = data

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class MyDataset(Dataset):
    def __init__(self, data: dict) -> None:
        self.data = data

    def __getitem__(self, index) -> Any:
        return {k: v[index] for k, v in self.data.items()}

    def __add__(self, other: MyDataset):
        now = self.data.copy()
        new = other.data.copy()
        for k in new.keys():
            if isinstance(new[k], list):
                try:
                    now[k] += new[k]
                except KeyError:
                    now[k] = new[k]
            else:
                try:
                    now[k] = torch.cat([now[k], new[k]])
                except KeyError:
                    now[k] = new[k]
        return MyDataset(now)

    def __len__(self) -> int:
        try:
            return len(self.data[list(self.data.keys())[0]])
        except:
            return 0

    def label_counts(self):
        return torch.bincount(self.data["labels"])

    def append(self, data: dict) -> None:
        for k, v in data.items():
            if not type(self.data[k]) is list:
                self.data[k] = torch.cat([self.data[k], v.unsqueeze(0)], dim=1)
            else:
                self.data[k].append(v)

    def split(self, proba: float = 0.5, shuffle: bool = True):
        pivot = int(self.__len__() * proba)
        indices = (
            torch.randperm(self.__len__())
            if shuffle
            else torch.tensor(list(range(self.__len__())))
        )
        return (self.subset(idx) for idx in [indices[:pivot], indices[pivot:]])

    def subset(self, indices: list) -> MyDataset:
        out = {}
        for k, v in self.data.items():
            if isinstance(v, list):
                out[k] = [v[idx] for idx in indices]
            else:
                out[k] = torch.cat([v[idx].unsqueeze(0) for idx in indices])
        return MyDataset(out)


class KafkaDataset(object):
    def __init__(
        self,
        topic: str,
        bootstrap_servers: List,
        batch_size: int = 16,
    ):
        self.consumer = KafkaConsumer(
            topic, bootstrap_servers=bootstrap_servers, auto_offset_reset="latest"
        )
        self.buffer = []
        self.batch_size = batch_size

    def __iter__(self):
        return self.listen()

    def listen(self) -> Any:
        for message in self.consumer:
            k = json.loads(message.key)
            v = json.loads(message.value)

            # add new to buffer
            self.buffer.append((k, v))

            # if buffer is ready, return buffer
            if len(self.buffer) >= self.batch_size:
                yield self.buffer
                self.buffer = []

            # simulate environment delay
            time.sleep(0.5)
