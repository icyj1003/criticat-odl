import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from avalanche.evaluation.metrics import MaxRAM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from buffer import FixedSizeBuffer, FullBuffer
from callbacks import EarlyStopping
from cusdataset import MyDataset
from metrics import MeanMetric
from module import *
from utils import *
from vocabulary import Vocabulary
from word2vec import PhoBertVector, PhoVector

# Paths
CUR_DIR = os.path.abspath(os.curdir)
CACHE_DIR = "D:/odl_cache/.cache/"
SESSION_DIR = "D:/odl_cache/.cache/sessions_256/"
CONFIG_DIR = os.path.join(CUR_DIR, "./configs/")
LOG_DIR = "D:/model_zoo/odl/"

# Settings
NUM_SESSIONS = len(os.listdir(SESSION_DIR)) - 1


def main():
    for filename in os.listdir(CONFIG_DIR):
        config_name = filename.split(".")[0]
        log_dir = os.path.join(LOG_DIR, config_name)

        # load configs
        configs = read_config(os.path.join(CONFIG_DIR, filename))

        # define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model from configs
        model = load_model_from_config(configs).to(device)

        # optimizer and criterion
        optimizer = optim.Adam(
            params=model.parameters(), lr=configs.get("learning_rate", 0.001)
        )
        criterion = nn.CrossEntropyLoss()

        # create vectorizer is ensemble of text
        vectorizer = None
        if configs["type"] == "ensemble" or configs["type"] == "text":
            if configs["text_model"].get("feature_extractor") == "bert":
                vectorizer = PhoBertVector(device=device)
            elif configs["text_model"].get("feature_extractor") == "pho":
                vectorizer = PhoVector(
                    name="word2vec_vi_words_100dims.txt", cache=CACHE_DIR
                )

        # metrics to measure catastrophic forgetting
        alpha_ideal = None
        base_loader = None
        omega_base = MeanMetric()
        omega_new = MeanMetric()
        omega_all = MeanMetric()
        ram_usage = MaxRAM()

        # memory buffer
        test_buffer = FullBuffer()  # to store all test data
        if configs.get("buffer") == "full":  # for full rehearsal
            buffer = FullBuffer()
        elif configs.get("buffer") == "fixed":  # for fixed size rehearsal
            buffer = FixedSizeBuffer(max_size=512, classes=2)
        else:
            buffer = None

        for session in range(NUM_SESSIONS):
            session_dir = os.path.join(log_dir, f"{session}")
            ram_usage.reset()
            ram_usage.start_thread()

            # load session data from cache
            cache = torch.load(os.path.join(SESSION_DIR, f"session_{session}.pt"))
            train_dict = cache.get("dataset")["train"]
            test_dict = cache.get("dataset")["test"]

            # extract word vectors if feature_extractor in [bert, pho]
            if vectorizer:
                train_dict["texts"] = torch.stack(
                    [
                        vectorizer.get_vecs_by_raw_text(raw)
                        for raw in train_dict["raw_texts"]
                    ]
                )
                test_dict["texts"] = torch.stack(
                    [
                        vectorizer.get_vecs_by_raw_text(raw)
                        for raw in test_dict["raw_texts"]
                    ]
                )

            # create dataset
            train_dataset = MyDataset(train_dict)
            test_dataset = MyDataset(test_dict)
            test_dataset, eval_dataset = test_dataset.split(shuffle=False)

            # create train_loader
            if buffer:
                buffer.append(train_dataset)
                train_loader = buffer.create_loader(
                    shuffle=False if session == 0 else True,
                    batch_size=configs.get("train_batch_size", 32),
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    shuffle=False,
                    batch_size=configs.get("train_batch_size", 32),
                )

            # create test_loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=configs.get("test_batch_size", 32),
            )

            # create all_loader
            test_buffer.append(test_dataset)
            all_loader = test_buffer.create_loader(
                batch_size=configs.get("test_batch_size", 32),
            )

            # create eval_loader
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=configs.get("eval_batch_size", 32),
            )

            # reset callbacks
            early_stopping = EarlyStopping(tolerance=5, min_delta=0, mode="auto")

            begin_time = time.time()
            for epoch in tqdm(
                range(configs.get("num_epochs")),
                desc=f"Training session {session}",
                unit="epochs",
            ):
                # train model on train_loader
                train_loss, train_acc, _, _ = train_epoch(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    loader=train_loader,
                    device=device,
                    configs=configs,
                )
                # eval model on eval_loader
                eval_loss, eval_acc, _, _ = eval_epoch(
                    model=model,
                    criterion=criterion,
                    loader=train_loader,
                    device=device,
                    configs=configs,
                )
                # print(
                #     f"Epoch: {epoch+1:03} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Eval loss: {eval_loss:.4f} | Eval acc: {eval_acc:.4f}"
                # )

                early_stopping(model, epoch, eval_loss, eval_acc)
                if early_stopping.early_stop:
                    break

            training_time = time.time() - begin_time

            # restore model from best epoch
            model.load_state_dict(early_stopping.weights)

            # save_object(model, session_dir, "best_model", mode="pt")

            # eval model on test_loader
            print("Evaluating session on current test set ...")
            _, alpha_new_i, _, _ = eval_epoch(
                model=model,
                criterion=criterion,
                loader=test_loader,
                device=device,
                configs=configs,
            )

            # base_loader
            if session == 0:
                # compute metrics
                alpha_ideal = alpha_new_i
                base_loader = test_loader
                ram_usage.stop_thread()
                print(
                    f"Session {session:02}:\nalpha_ideal: {alpha_ideal:.4f} | memory usage: {ram_usage.result()}"
                )

                # save results
                metrics = {
                    "alpha_ideal": alpha_ideal,
                    "training_time": training_time,
                    "ram_usage": ram_usage.result(),
                }
            else:
                # compute metrics
                # eval model on base_loader
                print("Evaluating session on base set ...")
                _, alpha_base_i, _, _ = eval_epoch(
                    model=model,
                    criterion=criterion,
                    loader=base_loader,
                    device=device,
                    configs=configs,
                )

                # eval model on all_loader
                print("Evaluating session on all seen sessions ...")
                _, alpha_all_i, _, _ = eval_epoch(
                    model=model,
                    criterion=criterion,
                    loader=all_loader,
                    device=device,
                    configs=configs,
                )

                # update metrics
                omega_base.update(alpha_base_i / alpha_ideal)
                omega_new.update(alpha_new_i)
                omega_all.update(alpha_all_i / alpha_ideal)
                ram_usage.stop_thread()
                print(
                    f"Session {session:02}:\nalpha_ideal: {alpha_ideal:.4f} | alpha_new_i: {alpha_new_i:.4f} | alpha_base_i: {alpha_base_i:.4f} | alpha_all_i: {alpha_all_i:.4f} | memory usage: {ram_usage.result()}"
                )
                metrics = {
                    "alpha_ideal": alpha_ideal,
                    "training_time": training_time,
                    "ram_usage": ram_usage.result(),
                    "alpha_new_i": alpha_new_i,
                    "alpha_base_i": alpha_base_i,
                    "alpha_all_i": alpha_all_i,
                }
            save_object(metrics, session_dir, "metric", mode="jsonc")
        print(
            f"omega_base: {omega_base.result():.4f} | omega_new: {omega_new.result():.4f} | omega_all: {omega_all.result():.4f}"
        )
        save_object(
            {
                "omega_base": omega_base.result(),
                "omega_new": omega_new.result(),
                "omega_all": omega_all.result(),
            },
            session_dir,
            "metric",
            mode="jsonc",
        )


if __name__ == "__main__":
    main()
