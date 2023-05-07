import torch

def accuracy(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy
