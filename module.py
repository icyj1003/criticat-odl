from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.transforms as TT
from torchvision import models
from transformers import AutoModel, AutoTokenizer, logging

logging.set_verbosity(logging.CRITICAL)


class Vocabulary:
    def __init__(self, specials: List[str] = None):
        self.w2i = {}
        self.i2w = {}
        self.idx = 0
        self.default_idx = None
        self.pad_idx = None

        if specials:
            self.append_tokens(specials)

    def append_tokens(self, tokens: List):
        for token in tokens:
            self.append_token(token)

    def append_token(self, token: str):
        if token not in self.w2i:
            self.w2i[token] = self.idx
            self.i2w[self.idx] = token
            self.idx += 1

    def append_from_iterator(self, iterator: Iterable):
        for tokens in iterator:
            for token in tokens:
                try:
                    self.append_token(token)
                except Exception as e:
                    pass

    def set_default_idx(self, idx: int):
        self.default_idx = idx

    def set_pad_idx(self, idx: int):
        self.pad_idx = idx

    def get_idxs(self, tokens: List):
        return torch.tensor([self[token] for token in tokens])

    def __contains__(self, token: str):
        return token in self.w2i

    def __getitem__(self, token: str):
        return self.w2i[token] if token in self.w2i else self.default_idx

    def __len__(self):
        return len(self.w2i)

    def __repr__(self) -> Any:
        return repr(self.w2i)


class BertEmbedding(nn.Module):
    def __init__(
        self, path: str = "vinai/phobert-base-v2", max_length=None, device="cuda"
    ):
        super(BertEmbedding, self).__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.bert = AutoModel.from_pretrained(path)

    def forward(self, sentences):
        # encode raw text with bert tokenizer
        encoded = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        # get input tokens and masks
        input_ids, attention_masks = (
            torch.tensor(encoded["input_ids"]).to(self.device),
            torch.tensor(encoded["attention_mask"]).to(self.device),
        )

        # get tokens embedding
        embedded = self.bert(input_ids, attention_masks)[0]

        return embedded


class TokenEmbedding(nn.Module):
    def __init__(
        self, max_length=None, embedding_dim=100, max_features=100000, device="cuda"
    ):
        super(TokenEmbedding, self).__init__()

        self.max_length = max_length
        self.device = device

        # create vocab
        self.vocab = Vocabulary(specials=["<pad>", "<unk>"])
        self.vocab.set_default_idx(self.vocab["<unk>"])
        self.vocab.set_pad_idx(self.vocab["<pad>"])

        # padding layer
        self.pad = TT.PadTransform(max_length=max_length, pad_value=self.vocab.pad_idx)

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=max_features,
            embedding_dim=embedding_dim,
            padding_idx=self.vocab.pad_idx,
        )

    def forward(self, sentences):
        vectors = []
        for sen in sentences:
            tokens = sen.split()[: self.max_length]
            if self.training:  # update new tokens while training
                self.vocab.append_tokens(tokens)
            vectors.append(self.pad(self.vocab.get_idxs(tokens)))
        vectors = torch.stack(vectors).to(self.device)
        return self.embedding(vectors)


class BiLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_classes=2,
        dropout_rate=0.1,
    ):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, embedded):
        # lstm pass
        out, _ = self.lstm(embedded)

        # dropout pass
        out = self.dropout(out[:, -1, :])

        # get logits
        out = self.fc(out)

        return out


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=2,
        dropout_rate=0.1,
    ):
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes]
        )
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded):
        # convolutional layer expects 4D input, so we unsqueeze the tensor
        embedded = embedded.unsqueeze(1)

        # apply convolutional filters
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # apply max-over-time pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))

        # get logits
        logits = self.fc(cat)

        return logits


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # first layer pass
        out1 = torch.relu(self.fc1(inputs))

        # second layer pass
        out2 = self.dropout(out1)

        # get logits
        logits = self.fc2(out2)

        return logits


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        model = models.vgg16(weights="VGG16_Weights.DEFAULT")
        model.classifier = model.classifier[:-1]

    def forward(self, inputs):
        return self.model(inputs)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model = nn.Sequential(
            *list(models.resnet18(weights="ResNet18_Weights.DEFAULT").children())[:-1]
        )

    def forward(self, inputs):
        return self.model(inputs)


# class Ensemble(nn.Module):
#     def __init__(
#         self,
#         text_model: nn.Module,
#         image_model: nn.Module,
#         user_model: nn.Module,
#         metadata_model: nn.Module,
#         num_classes: int = 2,
#     ):
#         super(Ensemble, self).__init__()
#         self.text_model = text_model
#         self.image_model = image_model
#         self.user_model = user_model
#         self.metadata_model = metadata_model
#         self.weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)

#     def forward(self, text, image, user_name, metadata):
#         out_text = self.text_model(text)
#         out_image = self.image_model(image)
#         out_user = self.user_model(user_name)
#         out_metadata = self.metadata_model(metadata)
#         out = (
#             out_text * self.weights[0]
#             + out_image * self.weights[1]
#             + out_user * self.weights[2]
#             + out_metadata * self.weights[3]
#         ) / self.weights.sum()

#         return out
