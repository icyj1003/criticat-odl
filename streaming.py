import argparse
import copy
import os
import time
from typing import *

import torch

from buffer import create_buffer_from_setting
from customdataset import TestingDataset, load_dataset_from_cache
from module import *
from train_tools import eval, extract_features, train_offline, train_online
from utils import (
    create_namespace,
    ensure_dir_exists,
    load_config_from_file,
    save_json,
    set_all_seeds,
)

parser = argparse.ArgumentParser()

default_input_fields = ["post_message", "image", "metadata", "user_name"]


def sum_dataset(dataset_list: List[TestingDataset]) -> TestingDataset:
    result = TestingDataset([])
    for dataset in dataset_list:
        result += dataset
    return result


def eval_then_save(
    F,
    G,
    test_dataset,
    input_fields,
    save_dir,
    filename,
    loss_function,
    device,
    batch_size,
):
    (
        _,
        test_acc,
        test_f1,
        test_precision,
        test_recall,
        _,
    ) = eval(
        test_dataset=test_dataset,
        input_fields=input_fields,
        G=G,
        F=F,
        batch_size=batch_size,
        loss_function=loss_function,
        device=device,
    )
    save_json(
        dir=save_dir,
        filename=filename,
        object={
            "acc": test_acc,
            "f1": test_f1,
            "precision": test_precision,
            "recall": test_recall,
        },
    )

    print(
        f"{filename}: acc: {test_acc:.4f} | f1: {test_f1:.4f} | precision: {test_precision:.4f} | recall: {test_recall:.4f}"
    )


def restore_from_checkpoint(dir):
    model_checkpoint = torch.load(os.path.join(dir, "checkpoint_best.pt"))
    G = model_checkpoint["G"]
    F = model_checkpoint["F"]

    cache = torch.load(os.path.join(dir, "cache.pt"))
    base_dataset = cache["base_dataset"]
    features = cache["features"]
    labels = cache["labels"]

    return G, F, base_dataset, features, labels


def base_initialize(args):
    print("Begin base initialization")

    # create models
    G, F, input_fields = create_models(args)

    # create optimizer
    if len(input_fields) != 1:
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": list(child_G.parameters())
                    + list(child_F.parameters()),
                    "lr": args.lr[i],
                }
                for i, child_G, child_F in zip(
                    range(len(input_fields)), G.list_G, F.list_F
                )
            ],
            lr=1e-2,
        )
    else:
        optimizer = torch.optim.Adam(
            params=list(G.parameters()) + list(F.parameters()),
            lr=args.lr[0],
        )

    train_dataset, test_dataset = load_dataset_from_cache(
        os.path.join(args.dataset_dir, "dataset_01.pt"),
        input_fields=input_fields,
    )

    # save dir
    save_dir = os.path.join(
        args.save_dir,
        args.name,
        f"base",
    )
    ensure_dir_exists(save_dir)

    # train model offline
    start_training = time.time()
    train_offline(
        train_dataset=train_dataset,
        dev_dataset=test_dataset,
        input_fields=input_fields,
        G=G,
        F=F,
        epochs=args.epochs,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        checkpoint_path=save_dir,
        patience=args.patience,
        delta=args.delta,
        verbose=args.verbose,
        device=args.device,
    )
    training_time = time.time() - start_training

    # eval model
    eval_then_save(
        F=F,
        G=G,
        test_dataset=test_dataset,
        input_fields=input_fields,
        save_dir=save_dir,
        filename="metrics.json",
        loss_function=torch.nn.CrossEntropyLoss(),
        device=args.device,
        batch_size=args.batch_size,
    )

    # extract features for buffer
    features, labels = extract_features(
        train_dataset, input_fields, G, args.batch_size, args.device
    )

    # save features
    torch.save(
        {
            "features": features,
            "labels": labels,
            "base_dataset": test_dataset,
        },
        os.path.join(save_dir, "cache.pt"),
    )

    # save metrics
    print(f"system.json: train time: {training_time}")

    save_json(
        save_dir,
        "system.json",
        {
            "training_time": training_time,
        },
    )


def streaming(args, buffer_setting):
    print("Training online with buffer type: ", buffer_setting["type"])
    print("Restore base cache")
    # load base checkpoint
    (
        G_state_dict,
        F_state_dict,
        base_dataset,
        features,
        labels,
    ) = restore_from_checkpoint(
        os.path.join(
            args.save_dir,
            args.name,
            f"base",
        )
    )
    previous_dataset = copy.deepcopy(base_dataset)

    # create model
    G, F, input_fields = create_models(args)
    G.load_state_dict(G_state_dict)
    F.load_state_dict(F_state_dict)

    # create buffer
    buffer = create_buffer_from_setting(
        buffer_setting, input_fields, features, labels, args.device
    )

    # create optimizer
    if len(input_fields) != 1:
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": list(child_F.parameters()),
                    "lr": args.lr[i],
                }
                for i, child_G, child_F in zip(
                    range(len(input_fields)), G.list_G, F.list_F
                )
            ],
            lr=1e-2,
        )
    else:
        optimizer = torch.optim.Adam(
            params=list(F.parameters()),
            lr=args.lr[0],
        )

    # loop over sessions
    for session_id, cache_file in enumerate(os.listdir(args.dataset_dir)):
        if cache_file == "dataset_01.pt":
            continue

        print(f"Loading {cache_file}")
        # load session dataset
        (train_dataset, test_dataset) = load_dataset_from_cache(
            os.path.join(args.dataset_dir, cache_file),
            input_fields=input_fields,
        )
        previous_dataset = previous_dataset + test_dataset

        # save dir
        save_dir = os.path.join(
            args.save_dir,
            args.name,
            buffer_setting["type"]
            if buffer_setting["type"] is not None
            else "no_buffer",
            f"s_{session_id + 1:02}",
        )
        ensure_dir_exists(save_dir)

        print(f"Training {cache_file}")
        start_training = time.time()
        # train model online
        train_online(
            online_dataset=train_dataset,
            input_fields=input_fields,
            G=G,
            F=F,
            optimizer=optimizer,
            loss_function=torch.nn.CrossEntropyLoss(),
            device=args.device,
            buffer=buffer,
            buffer_size=args.buffer_size,
        )
        training_time = time.time() - start_training

        # eval on base dataset
        eval_then_save(
            F,
            G,
            base_dataset,
            input_fields,
            save_dir,
            "base.json",
            torch.nn.CrossEntropyLoss(),
            args.device,
            args.batch_size,
        )

        # eval on new dataset
        eval_then_save(
            F,
            G,
            test_dataset,
            input_fields,
            save_dir,
            "new.json",
            torch.nn.CrossEntropyLoss(),
            args.device,
            args.batch_size,
        )

        # eval on all seen dataset
        eval_then_save(
            F,
            G,
            previous_dataset,
            input_fields,
            save_dir,
            "all.json",
            torch.nn.CrossEntropyLoss(),
            args.device,
            args.batch_size,
        )

        # save metrics
        buffer_length = len(buffer) if buffer else 0
        buffer_memory = buffer.storage_size() if buffer else 0

        print(
            f"system.json: train time: {training_time} | buffer length {buffer_length} | buffer memory: {buffer_memory:.4f} mb"
        )

        save_json(
            save_dir,
            "system.json",
            {
                "training_time": training_time,
                "buffer_length": buffer_length,
                "buffer_memory": buffer_memory,
            },
        )

    torch.save(
        {
            "G": G.state_dict(),
            "F": F.state_dict(),
            "optimizer": optimizer.state_dict(),
            "buffer": buffer
            if buffer_setting["type"] != "remind"
            else {
                "centroids": buffer.centroids,
                "inputs": buffer.inputs,
                "labels": buffer.labels,
            },
        },
        os.path.join(save_dir, "checkpoint_online.pt"),
    )


if __name__ == "__main__":
    parser.add_argument(
        "--setting_file",
        type=str,
        help="Path to the setting file",
        default=None,
    )

    parser.add_argument(
        "--setting_dir",
        type=str,
        help="Dir of setting files",
        default=None,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory of dataset",
        default=None,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the results",
        default=None,
    )

    parser.add_argument(
        "--base",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
    )

    file_path = parser.parse_args()

    assert file_path.setting_file != file_path.setting_dir

    if file_path.setting_dir is not None:
        for setting_file in os.listdir(file_path.setting_dir):
            if setting_file.endswith(".jsonc"):
                # load config
                configs = load_config_from_file(
                    os.path.join(file_path.setting_dir, setting_file)
                )

                # overwrite save_dir and dataset_dir
                if file_path.save_dir != None:
                    configs["save_dir"] = file_path.save_dir

                # create namespace
                if file_path.dataset_dir != None:
                    configs["dataset_dir"] = file_path.dataset_dir

                args = create_namespace(configs)

                # set seeds
                set_all_seeds(args.seed)

                if file_path.base:
                    base_initialize(args)

                if file_path.stream:
                    for bs in args.buffer_setting:
                        streaming(args, bs)
                    os.rename(
                        os.path.join(file_path.setting_dir, setting_file),
                        os.path.join(
                            file_path.setting_dir,
                            setting_file.replace(".jsonc", ".ok"),
                        ),
                    )

    elif file_path.setting_file is not None:
        # load config
        configs = load_config_from_file(file_path.setting_file)

        # overwrite save_dir and dataset_dir
        if file_path.save_dir != None:
            configs["save_dir"] = file_path.save_dir

        if file_path.dataset_dir != None:
            configs["dataset_dir"] = file_path.dataset_dir

        # create namespace
        args = create_namespace(configs)

        # set seeds
        set_all_seeds(args.seed)

        if file_path.base:
            base_initialize(args)

        if file_path.stream:
            for bs in args.buffer_setting:
                streaming(args, bs)
            os.rename(
                file_path.setting_file,
                file_path.setting_file.replace(".jsonc", ".ok"),
            )
