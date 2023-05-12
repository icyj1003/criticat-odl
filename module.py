import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#! === Text model ===
class LSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_classes=2, dropout_rate=0.5
    ):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        out, _ = self.lstm(embedded)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return F.softmax(out, dim=1)


class GRUClassifier(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_classes=2, dropout_rate=0.5
    ):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        out, _ = self.gru(embedded)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return F.softmax(out, dim=1)


#! === Image model ===
class Resnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, inputs):
        out = self.model(inputs)
        return F.softmax(out, dim=1)


class Resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34, self).__init__()
        self.model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, inputs):
        out = self.model(inputs)
        return F.softmax(out, dim=1)


#! === Metadata model ===
class Metadata(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(Metadata, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return F.softmax(out, dim=1)


#! === User Embedding ===
class UserEmbedding(nn.Module):
    def __init__(self, max_num_user, embedding_dim, num_classes=2, dropout_rate=0.5):
        super(UserEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_num_user, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        out = self.dropout(embedded)
        out = self.fc(out)
        return F.softmax(out, dim=1)


#! === Ensemble model ===
class Ensemble(nn.Module):
    def __init__(
        self,
        text_model: nn.Module,
        image_model: nn.Module,
        user_model: nn.Module,
        metadata_model: nn.Module,
        num_classes: int = 2,
    ):
        super(Ensemble, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.user_model = user_model
        self.metadata_model = metadata_model
        self.weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)

    def forward(self, text, image, user_name, metadata):
        out_text = self.text_model(text)
        out_image = self.image_model(image)
        out_user = self.user_model(user_name)
        out_metadata = self.metadata_model(metadata)
        out = (
            out_text * self.weights[0]
            + out_image * self.weights[1]
            + out_user * self.weights[2]
            + out_metadata * self.weights[3]
        ) / self.weights.sum()

        return out
