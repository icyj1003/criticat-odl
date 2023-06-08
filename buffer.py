import copy
import random
import sys
import time
from typing import *

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader


class ProductQuantizer:
    def __init__(self, M=32, ksub=256):
        self.M = M
        self.nbits = int(np.log2(ksub))

    def train(self, vectors):
        """_summary_

        Args:
            vectors (_type_): input vectors with shape [batch_size, vector_size]
        """

        assert len(vectors.shape) == 2

        self.pq = faiss.ProductQuantizer(vectors.shape[1], self.M, self.nbits)
        start = time.time()
        self.pq.train(vectors)
        print("Completed in {} secs".format(time.time() - start))

    def encode(self, vectors):
        return torch.tensor(self.pq.compute_codes(vectors))

    def decode(self, encoded_vectors):
        return torch.tensor(self.pq.decode(encoded_vectors.numpy()))

    def storage_size(self):
        return sys.getsizeof(
            torch.tensor(
                faiss.vector_float_to_array(self.pq.centroids)
            ).untyped_storage()
        )


class Buffer:
    def __init__(
        self,
        input_fields: List[
            Literal["post_message", "image", "metadata", "user_name"]
        ],
        max_capacity: int = 99999999,
    ) -> None:
        self.max_capacity = max_capacity
        self.inputs = {k: [] for k in input_fields}
        self.input_fields = input_fields
        self.labels = []

    def mix(self, *args, **kwargs):
        pass

    def __len__(self, *args, **kwargs):
        return len(self.labels)

    def __getitem__(self, idx, *args, **kwargs):
        return self.inputs[idx], self.labels[idx]

    def is_empty(self):
        return len(self) == 0

    def storage_size(self, *args, **kwargs):
        pass


class SimpleBuffer(Buffer):
    def __init__(
        self,
        input_fields: List[
            Literal["post_message", "image", "metadata", "user_name"]
        ],
        max_capacity: int = 99999999,
    ) -> None:
        super().__init__(input_fields, max_capacity)

    def mix(
        self,
        in_sample: List[Any],
        in_label,
        old_samples: int = 32,
        device="cuda",
    ):
        sample = copy.deepcopy([field.cpu() for field in in_sample])
        label = copy.deepcopy(in_label)
        # create a mix of old samples and new sample [batch, feature size], [batch]
        old_inputs = {k: [] for k in self.input_fields}
        old_labels = []
        if old_samples > 0:
            # if multi-input list of [batch, feature size], [batch]
            if not self.is_empty():
                # create index of old samples
                idxs = torch.randperm(len(self))[:old_samples]
                old_inputs = {
                    k: [self.inputs[k][idx] for idx in idxs]
                    for k in self.input_fields
                }
                old_labels = [self.labels[idx] for idx in idxs]

            # insert new sample to mix
            for i, k in enumerate(self.input_fields):
                old_inputs[k].append(sample[i])

            old_labels.append(label)

        # insert to buffer
        if len(self) < self.max_capacity:
            for i, k in enumerate(self.input_fields):
                self.inputs[k].append(sample[i])
            self.labels.append(label)
        else:
            idx = random.randint(0, self.max_capacity - 1)
            for i, k in enumerate(self.input_fields):
                self.inputs[k][idx] = sample[i]
            self.labels[idx] = label

        if old_samples > 0:
            out_inputs = [
                torch.cat(old_inputs[k], dim=0).to(device)
                for k in self.input_fields
            ]
            out_labels = torch.cat([o.reshape(1) for o in old_labels]).to(
                device
            )

            return out_inputs, out_labels

    def storage_size(self):
        return (
            sum(
                [
                    sum([sys.getsizeof(i.untyped_storage()) for i in v])
                    for v in self.inputs.values()
                ]
            )
            / 1024**2
        )


class REMIND(Buffer):
    def __init__(
        self,
        input_fields: List[
            Literal["post_message", "image", "metadata", "user_name"]
        ],
        input_vectors: List[Any],
        pq_fields: List[Literal["post_message", "user_name"]],
        M: int = 8,
        ksub: int = 256,
        max_capacity: int = 99999999,
    ) -> None:
        super().__init__(input_fields, max_capacity)
        self.shape = {}
        self.PQ = {}
        self.centroids = {}
        self.sum_pq = 0
        self.pq_fields = pq_fields
        for i, input_field in enumerate(input_fields):
            if input_field in pq_fields:
                self.shape[input_field] = input_vectors[i].shape[1:]
                temp = self.flatten(input_vectors[i])
                self.PQ[input_field] = ProductQuantizer(M=M, ksub=ksub)
                self.PQ[input_field].train(temp)
                self.sum_pq += self.PQ[input_field].storage_size()
                self.centroids[input_field] = faiss.vector_float_to_array(
                    self.PQ[input_field].pq.centroids
                )

    def flatten(self, vector):
        dim0 = vector.shape[0]
        return vector.reshape(dim0, -1)

    def storage_size(self):
        sum_vector = sum(
            [
                sum([sys.getsizeof(i.untyped_storage()) for i in v])
                for v in self.inputs.values()
            ]
        )
        return (sum_vector + self.sum_pq) / 1024**2

    def mix(
        self,
        in_sample: List[Any],
        in_label,
        old_samples: int = 32,
        device="cuda",
    ):
        sample = copy.deepcopy([field.cpu() for field in in_sample])
        label = copy.deepcopy(in_label)
        # create a mix of old samples and new sample [batch, feature size], [batch]
        old_inputs = {k: [] for k in self.input_fields}
        old_labels = []
        if old_samples > 0:
            # if multi-input list of [batch, feature size], [batch]
            if not self.is_empty():
                # create index of old samples
                idxs = torch.randperm(len(self))[:old_samples]
                old_inputs = {
                    k: [self.inputs[k][idx] for idx in idxs]
                    for k in self.input_fields
                }

                for k in self.pq_fields:
                    if k in self.input_fields:
                        temp = (
                            self.PQ[k]
                            .decode(torch.cat(old_inputs[k], dim=0))
                            .reshape((-1,) + self.shape[k])
                        )
                        old_inputs[k] = [t.unsqueeze(0) for t in temp]
                old_labels = [self.labels[idx] for idx in idxs]

            # insert new sample to mix
            for i, k in enumerate(self.input_fields):
                old_inputs[k].append(sample[i])

            old_labels.append(label)

        # insert to buffer
        for i, k in enumerate(self.input_fields):
            if k in self.pq_fields:
                sample[i] = self.PQ[k].encode(self.flatten(sample[i]))
        if len(self) < self.max_capacity:
            for i, k in enumerate(self.input_fields):
                self.inputs[k].append(sample[i])
            self.labels.append(label)
        else:
            idx = random.randint(0, self.max_capacity - 1)
            for i, k in enumerate(self.input_fields):
                self.inputs[k][idx] = sample[i]
            self.labels[idx] = label

        if old_samples > 0:
            out_inputs = [
                torch.cat(old_inputs[k], dim=0).to(device)
                for k in self.input_fields
            ]
            out_labels = torch.cat([o.reshape(1) for o in old_labels]).to(
                device
            )
            return out_inputs, out_labels


def create_buffer_from_setting(setting, input_fields, features, labels, device):
    buffer = None
    if setting["type"] == "unlimit":
        print("Using unlimit buffer")
        buffer = SimpleBuffer(input_fields=input_fields)
        for feature, label in zip(features, labels):
            buffer.mix(
                [f.unsqueeze(0) for f in feature],
                label,
                old_samples=0,
                device=device,
            )
    elif setting["type"] == "limit":
        print("Using limit buffer with max capacity", setting["max_capacity"])
        buffer = SimpleBuffer(
            input_fields=input_fields, max_capacity=setting["max_capacity"]
        )
        for feature, label in zip(features, labels):
            buffer.mix(
                [f.unsqueeze(0) for f in feature],
                label,
                old_samples=0,
                device=device,
            )
    elif setting["type"] == "remind":
        print(
            "Using REMIND with max capacity",
            setting["max_capacity"],
            "M",
            setting["M"],
            "ksub",
            setting["ksub"],
        )
        temp = {k: [] for k in input_fields}
        for i, k in enumerate(input_fields):
            temp[k] = torch.stack([feature[i] for feature in features], dim=0)

        buffer = REMIND(
            input_fields=input_fields,
            pq_fields=["post_message", "image", "user_name"],
            input_vectors=list(temp.values()),
            max_capacity=setting["max_capacity"],
            M=setting["M"],
            ksub=setting["ksub"],
        )

        for feature, label in zip(features, labels):
            buffer.mix(
                [f.unsqueeze(0) for f in feature],
                label,
                old_samples=0,
                device=device,
            )
    return buffer
