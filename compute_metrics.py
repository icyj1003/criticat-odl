import os
from pathlib import Path

import pandas as pd

from utils import *


# function compute average metrics of list[dict]
def average_metrics(metrics, normalize: dict = None):
    if not normalize:
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = sum([m[key] for m in metrics]) / len(metrics)
        return avg_metrics
    else:
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = sum(
                [m[key] / normalize[key] for m in metrics]
            ) / len(metrics)
        return avg_metrics


def compute_metrics(model_name, model_dir):
    metrics_base = load_json(
        os.path.join(model_dir, model_name, "base"), "metrics.json"
    )
    system_base = load_json(
        os.path.join(model_dir, model_name, "base"), "system.json"
    )

    model_size = (
        Path(os.path.join(model_dir, model_name, "base", "checkpoint_best.pt"))
        .stat()
        .st_size
        / 1024**2
    )

    results = []

    for setting in os.listdir(os.path.join(model_dir, model_name)):
        if setting == "base":
            continue
        result = {
            "name": model_name,
            "model_size": model_size,
            "buffer_type": setting,
        }
        base = []
        all = []
        new = []
        system = []
        for session in os.listdir(os.path.join(model_dir, model_name, setting)):
            base.append(
                load_json(
                    os.path.join(model_dir, model_name, setting, session),
                    "base.json",
                )
            )
            all.append(
                load_json(
                    os.path.join(model_dir, model_name, setting, session),
                    "all.json",
                )
            )
            new.append(
                load_json(
                    os.path.join(model_dir, model_name, setting, session),
                    "new.json",
                )
            )
            system.append(
                load_json(
                    os.path.join(model_dir, model_name, setting, session),
                    "system.json",
                )
            )
        o_new = {k + "_new": v for k, v in average_metrics(new).items()}
        o_all = {
            k + "_all": v
            for k, v in average_metrics(all, normalize=metrics_base).items()
        }
        o_base = {
            k + "_base": v
            for k, v in average_metrics(base, normalize=metrics_base).items()
        }

        training_time = average_metrics([system_base] + system)
        s_system = average_metrics(system)

        for r in [o_new, o_all, o_base, training_time, s_system]:
            result.update(r)

        results.append(result)

    return results


model_dir = "E:\\tools\\new_odl\\checkpoint\\reintel2020\\"

for model in os.listdir(model_dir):
    model_name = model

    try:
        df = pd.DataFrame({})

        for metrics in compute_metrics(model_name, model_dir):
            df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

        df.to_csv(f"./results/{model_name}.csv", index=False)
    except Exception as e:
        print(e)
        continue

    

results = []

for file in os.listdir("./results"):
    if file.startswith("result"):
        continue

    try:
        results.append(pd.read_csv(os.path.join("./results", file)))
    except:
        continue

df = pd.concat(results, ignore_index=True)
df.sort_values(by=["f1_all", "f1_new", "f1_base"], inplace=True, ascending=False)
df.to_csv("./results/result.csv", index=False)