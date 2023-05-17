from typing import *

import torch
from sklearn.metrics import accuracy_score, f1_score


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


class MeanAccuracy(MeanMetric):
    def __init__(self):
        super(MeanAccuracy, self).__init__()

    def update(self, y_true, y_pred, weight=1):
        self.total += accuracy_score(y_true=y_true, y_pred=y_pred) * weight
        self.weights.append(weight)

    def result(self):
        return self.total / sum(self.weights)


class MeanF1Score(MeanMetric):
    def __init__(
        self,
        average: Literal["micro", "macro", "samples", "weighted", "binary"]
        | None = "binary",
    ):
        super(MeanF1Score, self).__init__()
        self.average = average

    def update(self, y_true, y_pred, weight=1):
        self.total += (
            f1_score(y_true=y_true, y_pred=y_pred, average=self.average) * weight
        )
        self.weights.append(weight)

    def result(self):
        return self.total / sum(self.weights)


if __name__ == "__main__":
    mean = MeanMetric()
    for i in range(100):
        mean.update(i)
        print(mean.result(), mean.total, mean.weights)
