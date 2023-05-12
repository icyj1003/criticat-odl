import os
import pickle

import torch
import torch.optim as optim
from avalanche.evaluation.metrics import Accuracy, LossMetric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from cusdataset import TestingDataset
from module import *
from utils import (
    handling_metadata,
    handling_text,
    handling_username,
    read_config,
    train_one,
)
from vocabulary import Vocabulary

# Paths
CUR_DIR = os.path.abspath(os.curdir)
CACHE_DIR = os.path.join(CUR_DIR, "./.cache/")
CONFIG_DIR = os.path.join(CUR_DIR, "./configs/")
MODEL_DIR = "D:/model_zoo/odl/"


def load_dataset(configs):
    # load dataset from cache
    cache = open(os.path.join(CACHE_DIR, "final.pkl"), "rb")
    data = pickle.load(cache)
    cache.close()
    # create torch dataset and dataloader
    dataset = TestingDataset(data)
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"])
    del dataset, data
    return dataloader


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # load configs
    configs = read_config("./configs/train.jsonc")

    # load array from cache
    dataloader = load_dataset(configs)

    # Init wandb
    wandb.init(project="odl", config=configs)

    # load model
    if configs["ensemble"]:
        # === text model ====
        # text_vocabulary
        vc = Vocabulary(specials=["<pad>", "<unk>"])
        vc.set_default_idx(vc["<unk>"])

        # define model
        text_model_config = configs["text_model"]
        if text_model_config["architecture"] == "lstm":
            text_model = LSTMClassifier(
                vocab_size=text_model_config["vocab_size"],
                embedding_dim=text_model_config["embedding_dim"],
                hidden_dim=text_model_config["hidden_dim"],
            )
        elif text_model_config["architecture"] == "gru":
            text_model = GRUClassifier(
                vocab_size=text_model_config["vocab_size"],
                embedding_dim=text_model_config["embedding_dim"],
                hidden_dim=text_model_config["hidden_dim"],
            )
        else:
            raise Exception("Unknown text model architecture")

        # === image model ===
        image_model_config = configs["image_model"]
        if image_model_config["architecture"] == "resnet18":
            image_model = Resnet18()
        elif image_model_config["architecture"] == "resnet34":
            image_model = Resnet34()
        else:
            raise Exception("Unknown image model architecture")

        # === user model ===
        # user_vocabulary
        uvc = Vocabulary(specials=["<unk>"])
        uvc.set_default_idx(vc["<unk>"])
        user_model_config = configs["user_model"]
        if user_model_config["architecture"] == "embedding":
            user_model = UserEmbedding(
                max_num_user=100000, embedding_dim=100, num_classes=2
            )
        else:
            raise Exception("Unknown user model architecture")

        # === metadata model ===
        metadata_model_config = configs["metadata_model"]
        if metadata_model_config["architecture"] == "mlp":
            metadata_model = Metadata(
                input_size=6,
                hidden_size=metadata_model_config["hidden_size"],
                output_size=2,
            )
        else:
            raise Exception("Unknown metadata model architecture")

        # === ensemble model ===
        model = Ensemble(
            text_model=text_model,
            image_model=image_model,
            user_model=user_model,
            metadata_model=metadata_model,
        ).to(device)
    else:
        pass
    # optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # metrics
    total_acc = Accuracy()
    total_loss = LossMetric()

    # train
    for idx, batch in tqdm(enumerate(dataloader)):
        # * Text content
        texts = handling_text(batch["post_message"], vc).to(device)

        # * Username
        user_name = handling_username(batch["user_name"], uvc).to(device)

        # * Images
        images = batch["image"].to(device, dtype=torch.float)

        # * Metadata
        metadata = handling_metadata(
            num_like_post=batch["num_like_post"],
            num_comment_post=batch["num_comment_post"],
            num_share_post=batch["num_share_post"],
            raw_length=batch["raw_length"],
            timestamp_post=batch["timestamp_post"],
        ).to(device)

        # * Target
        labels = batch["label"].to(device)

        # * Train
        loss, outputs = train_one(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            inputs=(texts, images, user_name, metadata),
            targets=labels,
        )

        total_acc.update(outputs, labels)
        total_loss.update(loss, 1)

        wandb.log({"acc": total_acc.result(), "loss": total_loss.result()})

        del loss, outputs, texts, images, labels, user_name, metadata

    # save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(MODEL_DIR, "model.pt"),
    )


if __name__ == "__main__":
    main()
