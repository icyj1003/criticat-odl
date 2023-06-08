from __future__ import annotations

from collections import Counter
from typing import *

import torch

# from kafka import KafkaConsumer
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TestingDataset(Dataset):
    def __init__(self, data: List[dict] = []) -> None:
        self.data = data

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: TestingDataset) -> None:
        return TestingDataset(self.data + other.data)

    def ensure_keys(self, keys: List[str]) -> bool:
        return list(self.data[0].keys()) == keys

    def get_labels(self) -> List[int]:
        return [sample["label"] for sample in self.data]

    def subset(self, indices: List[int]) -> TestingDataset:
        return TestingDataset([self.data[i] for i in indices])

    def append(self, sample: dict):
        if self.ensure_keys(list(sample.keys())):
            self.data.append(sample)
        else:
            raise Exception(
                "Sample's keys do not match with the keys in the dataset"
            )

    def split(self, proba: float = 0.5, shuffle: bool = True):
        pivot = int(self.__len__() * proba)
        indices = (
            torch.randperm(self.__len__())
            if shuffle
            else list(range(self.__len__()))
        )
        return (self.subset(idx) for idx in [indices[:pivot], indices[pivot:]])


def load_dataset_from_cache(
    dir: str,
    input_fields=List[
        Literal["post_message", "user_name", "metadata", "image"]
    ],
):
    dataset = torch.load(dir)
    train_dataset = TestingDataset(
        [
            {k: v for k, v in d.items() if k in (input_fields + ["label"])}
            for d in dataset["train"].data
        ]
    )
    test_dataset = TestingDataset(
        [
            {k: v for k, v in d.items() if k in (input_fields + ["label"])}
            for d in dataset["test"].data
        ]
    )
    return train_dataset, test_dataset


# class KafkaDataset(object):
#     def __init__(
#         self,
#         topic: str,
#         bootstrap_servers: List,
#         batch_size: int = 16,
#     ):
#         self.consumer = KafkaConsumer(
#             topic, bootstrap_servers=bootstrap_servers, auto_offset_reset="latest"
#         )
#         self.buffer = []
#         self.batch_size = batch_size

#     def __iter__(self):
#         return self.listen()

#     def listen(self) -> Any:
#         for message in self.consumer:
#             k = json.loads(message.key)
#             v = json.loads(message.value)

#             # add new to buffer
#             self.buffer.append((k, v))

#             # if buffer is ready, return buffer
#             if len(self.buffer) >= self.batch_size:
#                 yield self.buffer
#                 self.buffer = []

#             # simulate environment delay
#             time.sleep(0.5)
