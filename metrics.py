import torch

def accuracy(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy


class SimplePrequentialMetric:
    def __init__(self, values=0):
        self.values = values
        self.t = 0

    def update(self, value):
        self.values = (self.values * self.t + value) / (self.t + 1)
        self.t += 1

    def __repr__(self) -> str:
        return repr(self.values)