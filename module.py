import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        out, _ = self.lstm(embedded)
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        out, _ = self.gru(embedded)
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1)


class Resnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, inputs):
        out = self.model(inputs)
        return F.softmax(out, dim=1)

