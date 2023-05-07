import json
from typing import Any, List
from kafka import KafkaConsumer
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import time


class TestingDataset(Dataset):
    def __init__(self, data: List[dict]) -> None:
        self.data = data

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


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

