import copy
import os
from collections import Counter
from typing import *

from torch.utils.data import DataLoader

from customdataset import TestingDataset
from utils import *


class Buffer:
    def __init__(
        self,
        data: List[dict] = [],
        buffer_type: Literal["full", "fixed", "classes"] = "full",
        max_capacity: int = 1024,
        num_classes: int = 2,
    ) -> None:
        self.buffer = TestingDataset(data)
        self.type = buffer_type
        self.num_classes = num_classes
        self.max_capacity = max_capacity

    def __len__(self) -> int:
        return len(self.buffer.data)

    def append(self, dataset: TestingDataset):
        if self.type == "full":
            self.buffer += dataset
        elif self.type == "fixed":
            old = copy.deepcopy(self.buffer)
            self.buffer = copy.deepcopy(dataset) + old.subset(
                torch.randperm(len(old))[: self.max_capacity - len(dataset)]
            )
            del old
        elif self.type == "classes":
            # old = copy.deepcopy(self.buffer)
            # new = copy.deepcopy(dataset)

            # current_counts = torch.tensor(dataset.get_labels()).bincount()

            # print(current_counts)
            new = copy.deepcopy(dataset)
            old = copy.deepcopy(self.buffer)

            # idx in old
            old_idx = list(range(len(old)))
            old_labels = old.get_labels()

            # label counts in new
            label_indices = {k: [] for k in range(self.num_classes)}
            for idx, label in enumerate(old_labels):
                label_indices[label] = label_indices.get(label, []) + [idx]
            label_indices = dict(sorted(label_indices.items()))

            # calculate number of instances to be added
            counts = torch.tensor(new.get_labels()).bincount()
            proba = counts / counts.sum()
            reversed_proba = 1 / proba
            print(proba, reversed_proba)
            self.buffer = new

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True):
        return DataLoader(self.buffer, batch_size=batch_size, shuffle=shuffle)


def main():
    DATASET_DIR = "D:/storage/odl/cache/online_session/reintel2020/"
    buffer = Buffer(buffer_type="classes")
    for file in os.listdir(DATASET_DIR):
        print("Reading file: ", file)
        cache = torch.load(os.path.join(DATASET_DIR, file))
        train_dataset = cache["train"]
        buffer.append(train_dataset)
        print(len(train_dataset), len(buffer))
        del cache


if __name__ == "__main__":
    main()
