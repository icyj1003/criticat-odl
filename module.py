import argparse
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
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
        self.max_vocab = 100000

        if specials:
            self.append_tokens(specials)

    def append_tokens(self, tokens: List):
        for token in tokens:
            self.append_token(token)

    def append_token(self, token: str):
        if self.max_vocab > len(self):
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

    def from_stoi(stoi):
        temp = Vocabulary()
        temp.w2i = stoi
        temp.i2w = {v: k for k, v in stoi.items()}
        temp.idx = max(stoi.values()) + 1
        temp.set_default_idx(0)
        temp.set_pad_idx(0)
        return temp


class BertEmbedding(nn.Module):
    def __init__(
        self,
        path: str = "vinai/phobert-base-v2",
        max_length=None,
        device="cuda",
    ):
        super(BertEmbedding, self).__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.bert = AutoModel.from_pretrained(path)

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

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
        self,
        max_length=None,
        embedding_dim=100,
        max_features=100000,
        pretrain_vectors=None,
        stoi=None,
        device="cuda",
    ):
        super(TokenEmbedding, self).__init__()

        self.max_length = max_length
        self.device = device
        if pretrain_vectors == None:
            # create vocab
            self.vocab = Vocabulary(specials=["<pad>", "<unk>"])
            self.vocab.set_default_idx(self.vocab["<unk>"])
            self.vocab.set_pad_idx(self.vocab["<pad>"])
            self.vocab.max_vocab = max_features

            # padding layer
            self.pad = TT.PadTransform(
                max_length=max_length, pad_value=self.vocab.pad_idx
            )

            # embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=max_features,
                embedding_dim=embedding_dim,
                padding_idx=self.vocab.pad_idx,
            )
        else:
            # create vocab
            self.vocab = Vocabulary.from_stoi(stoi=stoi)
            self.vocab.set_default_idx(0)
            self.vocab.set_pad_idx(0)
            self.vocab.max_vocab = max_features

            # padding layer
            self.pad = TT.PadTransform(
                max_length=max_length, pad_value=self.vocab.pad_idx
            )

            # embedding layer
            self.embedding = nn.Embedding.from_pretrained(pretrain_vectors)

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, sentences):
        vectors = []
        for sen in sentences:
            tokens = sen.split()[: self.max_length]
            if self.training:  # update new tokens while training
                self.vocab.append_tokens(tokens)
            vectors.append(self.pad(self.vocab.get_idxs(tokens)))
        vectors = torch.stack(vectors).to(self.device)

        embed = self.embedding(vectors)
        return embed


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


class BiGRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_classes=2,
        dropout_rate=0.1,
    ):
        super(BiGRU, self).__init__()

        self.gru = nn.GRU(
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
        out, _ = self.gru(embedded)

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
            [
                nn.Conv2d(1, num_filters, (fs, embedding_dim))
                for fs in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded):
        # convolutional layer expects 4D input, so we unsqueeze the tensor
        embedded = embedded.unsqueeze(1)

        # apply convolutional filters
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # apply max-over-time pooling
        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]

        # concatenate pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))

        # get logits
        logits = self.fc(cat)

        return logits


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_classes=2, dropout_rate=0.1
    ):
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
    def __init__(self, device="cuda"):
        super(VGG16, self).__init__()
        self.model = models.vgg16(weights="VGG16_Weights.DEFAULT")
        self.model.classifier = self.model.classifier[:-1]
        self.device = device

    def forward(self, inputs):
        feature = self.model(inputs.to(self.device))
        return feature  # [batch_size, 4092]


class Resnet18(nn.Module):
    def __init__(self, device="cuda"):
        super(Resnet18, self).__init__()
        self.model = nn.Sequential(
            *list(
                models.resnet18(weights="ResNet18_Weights.DEFAULT").children()
            )[:-1]
        )
        self.device = device

    def forward(self, inputs):
        feature = torch.flatten(self.model(inputs.to(self.device)), start_dim=1)
        return feature  # [batch_size, 512]
    
class Resnet50(nn.Module):
    def __init__(self, device="cuda"):
        super(Resnet50, self).__init__()
        self.model = nn.Sequential(
            *list(
                models.resnet50(weights="ResNet50_Weights.DEFAULT").children()
            )[:-1]
        )
        self.device = device

    def forward(self, inputs):
        feature = torch.flatten(self.model(inputs.to(self.device)), start_dim=1)
        return feature  # [batch_size, 512]


class UserEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim=300,
        max_features=100000,
        device="cuda",
    ):
        super(UserEmbedding, self).__init__()

        self.device = device

        # create vocab
        self.vocab = Vocabulary(specials=["<unk>"])
        self.vocab.set_default_idx(self.vocab["<unk>"])

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=max_features,
            embedding_dim=embedding_dim,
        )

    def freeze(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, user_names):
        outs = []
        for user_name in user_names:
            if self.training:  # update new tokens while training
                self.vocab.append_tokens(user_name)
            outs.append(torch.tensor(self.vocab[user_name]))
        outs = torch.stack(outs).to(self.device)
        return self.embedding(outs)


class MultiG(nn.Module):
    def __init__(self, list_G: List[nn.Module], device="cuda"):
        super(MultiG, self).__init__()
        self.list_G = list_G
        self.device = device

    def forward(self, *inputs):
        encoded_inputs = []
        for G, input in zip(self.list_G, inputs):
            if G is not None:
                encoded_inputs.append(G(input))
            else:
                encoded_inputs.append(input.to(self.device))

        return encoded_inputs


class MultiF(nn.Module):
    def __init__(
        self,
        list_F: List[nn.Module],
    ):
        super(MultiF, self).__init__()
        self.list_F = list_F
        self.weights = nn.Parameter(
            torch.ones(len(list_F)) / len(list_F), requires_grad=True
        )

    def forward(self, *encoded_inputs):
        outs = [
            F(encoded_input)
            for F, encoded_input in zip(self.list_F, encoded_inputs)
        ]
        return (
            sum([self.weights[i] * out for i, out in enumerate(outs)])
            / self.weights.sum()
        )


class DummyG(nn.Module):
    def __init__(self, device):
        super(DummyG, self).__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)


def create_models(args):
    """Create G and F models from args."""
    G = []
    F = []
    input_fields = []

    device = args.device
    input_fields = list(args.models.keys())

    for model_type in input_fields:
        temp_G, temp_F = create_SA_model(
            model_type, args.models[model_type], device
        )
        G.append(temp_G)
        F.append(temp_F)

    if len(args.models) == 1:
        print(f"Stand-alone {input_fields[0]} model")
        return G[0], F[0], input_fields
    else:
        print(f"Multimodal with features order: {' -> '.join(input_fields)}")
        return (
            MultiG(G, device=device).to(device),
            MultiF(F).to(device),
            input_fields,
        )


def create_SA_model(model_type, setting, device):
    G = DummyG(device)
    F = None
    # ! Metadata
    if model_type == "metadata":
        F = MLP(
            input_size=6,
            hidden_size=setting["hidden_dim"],
            dropout_rate=setting["dropout"],
        ).to(device)
    # ! Post message
    elif model_type == "post_message":
        if setting["G"] == "train":
            G = TokenEmbedding(
                max_length=setting["max_length"],
                embedding_dim=setting["embed_dim"],
                max_features=200000,
                device=device,
            ).to(device)
        elif setting["G"] == "bert":
            G = BertEmbedding(
                max_length=setting["max_length"], device=device
            ).to(device)
        elif setting["G"] == "pho":
            vector = torchtext.vocab.Vectors(
                cache="./.vector_cache/",
                name="word2vec_vi_words_300dims.txt",
                max_vectors=setting["max_vocab_size"],
            )
            G = TokenEmbedding(
                max_length=setting["max_length"],
                embedding_dim=setting["embed_dim"],
                pretrain_vectors=vector.vectors,
                stoi=vector.stoi,
                device=device,
            ).to(device)

        if setting["F"] == "bilstm":
            F = BiLSTM(
                hidden_dim=setting["hidden_dim"],
                embedding_dim=setting["embed_dim"],
                dropout_rate=setting["dropout"],
            ).to(device)
        if setting["F"] == "bigru":
            F = BiGRU(
                hidden_dim=setting["hidden_dim"],
                embedding_dim=setting["embed_dim"],
                dropout_rate=setting["dropout"],
            ).to(device)
        elif setting["F"] == "textcnn":
            F = TextCNN(
                embedding_dim=setting["embed_dim"],
                num_filters=setting["num_filters"],
                filter_sizes=setting["filter_sizes"],
            ).to(device)

    # ! User name
    elif model_type == "user_name":
        G = UserEmbedding(
            setting["embed_dim"],
            max_features=setting["max_users"],
            device=device,
        ).to(device)
        F = MLP(
            input_size=setting["embed_dim"],
            hidden_size=setting["hidden_dim"],
            dropout_rate=setting["dropout"],
        ).to(device)
    elif model_type == "image":
        if setting["G"] == "vgg16":
            G = VGG16(device=device).to(device)
            F = MLP(
                input_size=4096,
                hidden_size=setting["hidden_dim"],
                dropout_rate=setting["dropout"],
            ).to(device)
        elif setting["G"] == "resnet18":
            G = Resnet18(device=device).to(device)
            F = MLP(
                input_size=512,
                hidden_size=setting["hidden_dim"],
                dropout_rate=setting["dropout"],
            ).to(device)
        elif setting["G"] == "resnet50":
            G = Resnet50(device=device).to(device)
            F = MLP(
                input_size=2048,
                hidden_size=setting["hidden_dim"],
                dropout_rate=setting["dropout"],
            ).to(device)

    return G, F
