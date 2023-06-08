import itertools
from typing import *

from utils import save_json

# const
INPUT_FIELDS = ["post_message", "image", "metadata", "user_name"]

TEXT_HIDDEN_DIM = 256
TEXT_CNN_NUM_FILTERS = 32
TEXT_CNN_FILTER_SIZES = [2, 3, 4]
TEXT_MAX_LENGTH = 64
TEXT_MAX_VOCAB_SIZE = 1000000
TEXT_DROPOUT = 0.1
TEXT_G = ["train", "bert", "pho"]
TEXT_F = ["bilstm", "textcnn", "bigru"]

IMAGE_HIDDEN_DIM = 256
IMAGE_DROPOUT = 0.1
IMAGE_G = ["resnet18", "resnet50"]
IMAGE_F = ["MLP"]

METADATA_HIDDEN_DIM = 256
METADATA_DROPOUT = 0.1
METADATA_G = ["None"]
METADATA_F = ["MLP"]

USER_NAME_HIDDEN_DIM = 128
USER_NAME_DROPOUT = 0.1
USER_NAME_MAX_USERS = 100000
USER_NAME_G = ["embedding"]
USER_NAME_F = ["MLP"]

EPOCHS = 50


def create_name(models: List[dict]):
    name = ""
    name += "multimodal_" if len(models) > 1 else "standalone_"
    name += "_".join(
        [
            model.get("G", "None") + "+" + model.get("F", "MLP")
            for model in models
        ]
    )
    return name


def create_model():
    models_list = []
    lr_list = []
    inputs = []
    for r in range(1, 5):
        if r != 3:
            inputs += list(itertools.combinations(INPUT_FIELDS, r))

    new_inputs = []

    for input in inputs:
        if len(input) == 1 or input[0] == "post_message":
            new_inputs.append(input)

    for setting in new_inputs:
        for textG in TEXT_G:
            for textF in TEXT_F:
                for imageG in IMAGE_G:
                    for imageF in IMAGE_F:
                        for metadataG in METADATA_G:
                            for metadataF in METADATA_F:
                                for userG in USER_NAME_G:
                                    for userF in USER_NAME_F:
                                        model = {}
                                        lrs = []
                                        if "post_message" in setting:
                                            model["post_message"] = {
                                                "G": textG,
                                                "F": textF,
                                                "max_length": TEXT_MAX_LENGTH,
                                                "embed_dim": 300
                                                if textG != "bert"
                                                else 768,
                                                "dropout": TEXT_DROPOUT,
                                                "max_vocab_size": TEXT_MAX_VOCAB_SIZE,
                                                "hidden_dim": TEXT_HIDDEN_DIM,
                                                "num_filters": TEXT_CNN_NUM_FILTERS,
                                                "filter_sizes": TEXT_CNN_FILTER_SIZES,
                                            }
                                            if textG == "bert":
                                                lrs.append(2e-5)
                                            else:
                                                lrs.append(1e-2)
                                        if "image" in setting:
                                            model["image"] = {
                                                "G": imageG,
                                                "F": imageF,
                                                "hidden_dim": IMAGE_HIDDEN_DIM,
                                                "dropout": IMAGE_DROPOUT,
                                            }
                                            lrs.append(1e-3)
                                        if "metadata" in setting:
                                            model["metadata"] = {
                                                "G": metadataG,
                                                "F": metadataF,
                                                "hidden_dim": METADATA_HIDDEN_DIM,
                                                "dropout": METADATA_DROPOUT,
                                            }
                                            lrs.append(1e-2)
                                        if "user_name" in setting:
                                            model["user_name"] = {
                                                "G": userG,
                                                "F": userF,
                                                "max_users": USER_NAME_MAX_USERS,
                                                "embed_dim": 128,
                                                "hidden_dim": USER_NAME_HIDDEN_DIM,
                                                "dropout": USER_NAME_DROPOUT,
                                            }
                                            lrs.append(1e-1)
                                        models_list.append(model)
                                        lr_list.append(lrs)

    return models_list, lr_list


if __name__ == "__main__":
    models_list, lr_list = create_model()

    for models, lrs in zip(models_list, lr_list):
        name = create_name(models.values())
        if name.startswith("standalone"):
            bs = 32
        elif (
            "post_message" in models and models["post_message"]["G"] == "bert"
        ) or "image" in models:
            bs = 16
        else:
            bs = 32
        setting = {
            "name": name,
            "verbose": True,
            "batch_size": bs,
            "epochs": 20,
            "lr": lrs,
            "early_stopping": True,
            "delta": 0.0001,
            "decay": 1e-4,
            "patience": 5,
            "device": "cuda:0",
            "seed": 42,
            "buffer_size": 20,
            "buffer_setting": [
                {"type": None},
                {"type": "unlimit"},
                {"type": "limit", "max_capacity": 512},
                {
                    "type": "remind",
                    "max_capacity": 512,
                    "M": 32,
                    "ksub": 256,
                },
            ],
            "dataset_dir": "E:\\tools\\new_odl\\cache\\online_session\\reintel2020\\",
            "save_dir": "E:\\tools\\new_odl\\checkpoint\\reintel2020\\",
            # "dataset_dir": "/content/drive/MyDrive/Projects/thesis/cache/online_session/reintel2020",
            # "save_dir": "/content/drive/MyDrive/Projects/thesis/checkpoint/reintel2020",
            "models": models,
        }
        save_json("./configs/", name + ".jsonc", setting)
