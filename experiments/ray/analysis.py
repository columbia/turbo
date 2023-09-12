import json
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import yaml

from turbo.utils.utils import LOGS_PATH


def load_ray_experiment(logs: Union[Path, str]) -> pd.DataFrame:
    results = []
    for run_result in logs.glob("**/result.json"):
        try:
            with open(run_result, "r") as f:
                d = json.load(f)
            results.append(d)
        except Exception:
            pass
    df = pd.DataFrame(results)
    return df


def load_latest_ray_experiment() -> pd.DataFrame:
    log_dirs = list(LOGS_PATH.joinpath("ray").iterdir())

    latest_log_dir = max(log_dirs, key=lambda x: x.name)

    # Noisy logs so we don't forget which directory we're using
    print(latest_log_dir)

    return load_ray_experiment(latest_log_dir)


def load_tasks(expname="", validate=False, tasks_dir="") -> pd.DataFrame:
    if not expname:
        exp_dirs = list(LOGS_PATH.glob("exp_*"))
        latest_exp_dir = max(exp_dirs, key=lambda x: x.name)
    else:
        latest_exp_dir = LOGS_PATH.joinpath(expname)
    d = defaultdict(list)

    for p in latest_exp_dir.glob("**/*.json"):
        with open(p) as f:
            run_dict = json.load(f)
        for t in run_dict["tasks"]:
            block_budget = list(t["budget_per_block"].values())[0]
            d["id"].append(t["id"])
            d["first_block_id"] = min(
                [int(block_id) for block_id in t["budget_per_block"].keys()]
            )
            d["n_blocks"].append(len(t["budget_per_block"]))
            d["profit"].append(t["profit"])
            d["creation_time"].append(t["creation_time"])

            # NOTE: scheduler dependent
            # d["scheduling_time"].append(t["scheduling_time"])
            # d["scheduling_delay"].append(t["scheduling_delay"])
            # d["allocated"].append(t["allocated"])
            d["nblocks_maxeps"].append(
                f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
            )
        if not validate:
            break
        else:
            raise NotImplementedError
    df = pd.DataFrame(d).sort_values("id")

    if tasks_dir:
        maxeps = {}
        for task_file in Path(tasks_dir).glob("*.yaml"):
            task_dict = yaml.safe_load(task_file.open("r"))
            maxeps[f"{task_dict['rdp_epsilons'][-1]:.3f}"] = task_file.stem
        maxeps

        def get_task_name(s):
            n, m = s.split("-")
            return f"{n}-{maxeps[m]}"

        df["task"] = df["nblocks_maxeps"].apply(get_task_name)

    return df
