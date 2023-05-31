from typing import *

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metric:
    def update(self, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def result(self, **kwargs):
        pass


class MeanMetric(Metric):
    def __init__(self, default_total=0):
        self.total = default_total
        self.weights = []

    def update(self, values, weight=1):
        self.total += values * weight
        self.weights.append(weight)

    def result(self):
        return self.total / sum(self.weights)

    def reset(self):
        self.__init__()


class Compose(MeanMetric):
    def __init__(self):
        super(Compose, self).__init__
        self.metrics = {
            "accuracy": MeanMetric(),
            "f1-macro": MeanMetric(),
            "recall-macro": MeanMetric(),
            "precision-macro": MeanMetric(),
        }

    def update(
        self,
        dict_metrics,
        normalize={
            "accuracy": 1,
            "f1-macro": 1,
            "recall-macro": 1,
            "precision-macro": 1,
        },
        weight=1,
    ):
        for k in dict_metrics.keys():
            self.metrics[k].update(dict_metrics[k] / normalize[k], weight)

    def result(self):
        return {k: v.result() for k, v in self.metrics.items()}

    def reset(self):
        for k, v in self.metrics.items():
            v.reset()


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1-macro": f1_score(y_true, y_pred, average="macro"),
        "recall-macro": recall_score(y_true, y_pred, average="macro"),
        "precision-macro": precision_score(y_true, y_pred, average="macro"),
    }
