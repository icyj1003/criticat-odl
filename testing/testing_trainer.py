import torch
from module import LSTMClassifier, GRUClassifier
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim
from preprocess import TextCleaner
from cusdataset import TextDatasetForClassification
from trainer import Trainer

CUR_PATH = os.path.abspath(os.curdir)
VNCORE_NLP_PATH = os.path.join(CUR_PATH, "./vncorenlp/")
STOPWORDS_PATH = os.path.join(CUR_PATH, "./stop_words/vietnamese-stopwords-dash.txt")
DATASET_PATH = os.path.abspath("./data/")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "vihsd/train.csv")
DEV_DATASET_PATH = os.path.join(DATASET_PATH, "vihsd/dev.csv")
TEST_DATASET_PATH = os.path.join(DATASET_PATH, "vihsd/test.csv")

EMBED_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 128
NUM_EPOCHS = 100
NUM_CLASSES = 3

cleaner = TextCleaner(
    cur_dir=CUR_PATH, stopwords_path=STOPWORDS_PATH, vncorenlp_path=VNCORE_NLP_PATH
)

train_dataset = TextDatasetForClassification(TRAIN_DATASET_PATH, cleaner=cleaner)
dev_dataset = TextDatasetForClassification(
    DEV_DATASET_PATH, cleaner=cleaner, vocab=train_dataset.vocab
)

train_iter = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
dev_iter = DataLoader(dev_dataset, shuffle=True, batch_size=BATCH_SIZE)

# Initialize the model, loss function, optimizer and callbacks
model = GRUClassifier(
    len(train_dataset.vocab), EMBED_DIM, HIDDEN_DIM, num_classes=NUM_CLASSES
)

trainer = Trainer(
    model=model,
    epochs=NUM_EPOCHS,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss(),
    train_iter=train_iter,
    eval_iter=dev_iter,
)

trainer.fit()