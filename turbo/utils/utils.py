import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import omegaconf
import pandas as pd
import scipy

# from turbo.utils.plot import plot_budget_utilization_per_block, plot_task_status
from turbo.budget.renyi_budget import RenyiBudget

CUSTOM_LOG_PREFIX = "custom_log_prefix"
REPO_ROOT = Path(__file__).parent.parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("turbo/config/default.json")

FAILED = "failed"
PENDING = "pending"
FINISHED = "finished"

LAPLACE_RUNTYPE = "Laplace"
HISTOGRAM_RUNTYPE = "Histogram"
PMW_RUNTYPE = "PMW"


def get_node_key(blocks: Tuple[int, int]) -> str:
    """For some reason logs are using strings.
    You can use this for cache keys too.
    """
    return str(blocks)


def mlflow_log(key, value, step):
    mlflow_run = mlflow.active_run()
    if mlflow_run:
        mlflow.log_metric(
            key,
            value,
            step=step,
        )


def parse_block_requests_pattern(block_requests_pattern, max_blocks=50):
    # Just a regular distribution
    if isinstance(
        block_requests_pattern, omegaconf.omegaconf.ListConfig
    ) and isinstance(block_requests_pattern[0], int):
        return block_requests_pattern

    # Parameters for a discrete Gaussian
    distribution, std, mean = block_requests_pattern.split("-")
    std, mean = int(std), int(mean)
    assert distribution == "dgaussian"
    # mean = tmax - 2*std
    f = np.array([scipy.stats.norm.pdf(k, mean, std) for k in range(1, max_blocks + 1)])

    # Truncate after 2 stdev
    f = f / scipy.stats.norm.pdf(2 * std, 0, std)
    f = np.floor(f)

    # Frequency encoded by repetition (yes)
    blocks = []
    for i in range(1, 51):
        blocks.extend([i] * int(f[i - 1]))
    return blocks


def satisfies_constraint(blocks, branching_factor=2):
    """
    Checks if <blocks> satisfies the binary structure constraint
    """
    n = blocks[1] - blocks[0] + 1
    if not math.log(n, branching_factor).is_integer():
        return False
    if (blocks[0] % n) != 0:
        return False
    return True


def get_blocks_size(blocks, blocks_metadata):
    if isinstance(blocks, tuple):
        if "block_size" in blocks_metadata:
            # All blocks have the same size
            num_blocks = blocks[1] - blocks[0] + 1
            n = num_blocks * blocks_metadata["block_size"]
        else:
            n = sum(
                [
                    float(blocks_metadata["blocks"][str(id)]["size"])
                    for id in range(blocks[0], blocks[1] + 1)
                ]
            )
        return n
    else:
        return float(blocks_metadata["blocks"][str(blocks)]["size"])


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def get_logs(
    tasks_info,
    block_budgets_info,
    config_dict,
    **kwargs,
) -> dict:

    n_allocated_tasks = 0
    total_histogram_runs = 0
    total_laplace_runs = 0
    total_sv_misses = 0
    total_sv_checks = 0

    # Finally logging only a small number of tasks for faster analysis
    chunks = {}
    tasks_to_log = []
    cumulative_budget_per_block = {}
    sv_misses = {}
    sv_checks = {}

    if not config_dict["puredp"]:
        blocks_initial_budget = RenyiBudget.from_epsilon_delta(
            epsilon=config_dict["budget_accountant"]["epsilon"],
            delta=config_dict["budget_accountant"]["delta"],
            alpha_list=config_dict["budget_accountant"]["alphas"],
        )
    else:
        blocks_initial_budget = config_dict["budget_accountant"]["epsilon"]

    for i, task_info in enumerate(tasks_info):

        if task_info["status"] == FINISHED:
            n_allocated_tasks += 1

            chunk_keys = list(task_info["run_metadata"]["run_types"][0].keys())
            for chunk_key in chunk_keys:
                if chunk_key not in chunks:
                    chunks[str(chunk_key)] = 0
                chunks[str(chunk_key)] += 1

            run_metadata = task_info["run_metadata"]
            histogram_runs = laplace_runs = 0
            run_types_list = run_metadata["run_types"]
            for run_types in run_types_list:
                for run_type in run_types.values():
                    if run_type == "Laplace":
                        laplace_runs += 1
                    elif run_type == "Histogram":
                        histogram_runs += 1

            task_info["laplace_runs"] = laplace_runs
            task_info["histogram_runs"] = histogram_runs

            total_laplace_runs += laplace_runs
            total_histogram_runs += histogram_runs

            sv_check_status_list = run_metadata["sv_check_status"]
            node_id_list = run_metadata["sv_node_id"]
            assert len(sv_check_status_list) <= 1
            for sv_check_status in sv_check_status_list:
                sv_node_id = str(node_id_list[0])
                if sv_check_status == False:
                    total_sv_misses += 1
                    if sv_node_id not in sv_misses:
                        sv_misses[sv_node_id] = 0
                    sv_misses[sv_node_id] += 1
                if sv_node_id not in sv_checks:
                    sv_checks[sv_node_id] = 0
                sv_checks[sv_node_id] += 1
                total_sv_checks += 1

            ##### Total task budget per block across all trials #####
            total_task_budget_per_block = {}
            budget_per_block_list = run_metadata["budget_per_block"]
            for budget_per_block in budget_per_block_list:
                for block, budget in budget_per_block.items():
                    if block not in total_task_budget_per_block:
                        total_task_budget_per_block[block] = budget
                    else:
                        total_task_budget_per_block[block] += budget
            #########################################################

            ##### Total budget per block up to this task/time #####
            for block, budget in total_task_budget_per_block.items():
                if block not in cumulative_budget_per_block:
                    cumulative_budget_per_block[block] = budget
                else:
                    cumulative_budget_per_block[block] += budget
            #########################################################

            # Convert budgets to serializable form and save them
            cumulative_budget_per_block_dump = {
                block: budget.dump()
                for block, budget in cumulative_budget_per_block.items()
            }
            total_task_budget_per_block_dump = {
                block: budget.dump()
                for block, budget in total_task_budget_per_block.items()
            }
            task_info["run_metadata"][
                "budget_per_block"
            ] = total_task_budget_per_block_dump
            task_info["run_metadata"][
                "cumulative_budget_per_block"
            ] = cumulative_budget_per_block_dump
            #########################################################

            # Final Global budget consumption across all blocks - to output in the terminal
            if i == len(tasks_info) - 1:
                # For each block find the cumulative total budget
                global_budget = 0
                for block, budget in cumulative_budget_per_block.items():
                    global_budget += budget.epsilon

            if (
                task_info["id"] % int(config_dict["logs"]["log_every_n_tasks"]) == 0
                or i == len(tasks_info) - 1
            ):
                tasks_to_log.append(task_info)

    workload = pd.read_csv(config_dict["tasks"]["path"])
    query_pool_size = len(workload["query_id"].unique())
    config = {}

    # Fix a key for each run
    (
        key,
        mechanism_type,
        heuristic,
        warmup,
        learning_rate,
        tau,
        external_update_on_cached_results,
    ) = set_run_key(config_dict)

    config.update(
        {
            "n_allocated_tasks": n_allocated_tasks,
            "total_tasks": len(tasks_info),
            "total_sv_misses": total_sv_misses,
            "total_sv_checks": total_sv_checks,
            "total_histogram_runs": total_histogram_runs,
            "total_laplace_runs": total_laplace_runs,
            "mechanism": mechanism_type,
            "planner": config_dict["planner"]["method"],
            "workload_path": config_dict["tasks"]["path"],
            "query_pool_size": query_pool_size,
            "tasks_info": tasks_to_log,
            "block_budgets_info": block_budgets_info,
            "blocks_initial_budget": blocks_initial_budget,
            "zipf_k": config_dict["tasks"]["zipf_k"],
            "heuristic": heuristic,
            "tau": tau,
            "direct_match_caching": config_dict["exact_match_caching"],
            "config": config_dict,
            "learning_rate": learning_rate,
            "warmup": warmup,
            "external_update_on_cached_results": external_update_on_cached_results,
            "global_budget": global_budget,
            "chunks": chunks,
            "sv_misses": sv_misses,
            "key": key,
        }
    )

    # Any other thing to log
    for key, value in kwargs.items():
        config[key] = value
    return config


def set_run_key(config_dict):
    # Fix a key for each run
    if config_dict["logs"].get("mlflow_random_prefix", False):
        # Short nickname to identify runs, especially on long names that get cut by Plotly
        key = str(uuid.uuid4())[:4]
    else:
        key = ""

    tau = ""

    exact_match_caching = config_dict["exact_match_caching"]
    if config_dict["mechanism"]["type"] == "Laplace":
        mechanism_type = "Laplace"
        heuristic = ""
        learning_rate = ""
        warmup = ""
        external_update_on_cached_results = ""
        if config_dict["planner"]["method"] == "NoCuts" and exact_match_caching == True:
            mechanism_type += "+Cache"
        if (
            config_dict["planner"]["method"] == "MinCuts"
            and exact_match_caching == True
        ):
            mechanism_type += "+TreeCache"
        if (
            config_dict["planner"]["method"] == "MaxCuts"
            and exact_match_caching == True
        ):
            mechanism_type += "+PerPartitionCache"
        key += mechanism_type

    elif config_dict["mechanism"]["type"] == "PMW":
        external_update_on_cached_results = ""
        mechanism_type = "PMW"
        heuristic = ""
        learning_rate = ""
        warmup = ""
        key += mechanism_type

    # elif config_dict["mechanism"]["type"] == "TimestampsPMW":
    #     mechanism_type = "TimestampsPMW"
    #     heuristic = ""
    #     learning_rate = ""
    #     warmup = ""
    #     key += mechanism_type

    else:
        mechanism_type = "Hybrid"
        warmup = str(config_dict["mechanism"]["probabilistic_cfg"]["bootstrapping"])
        heuristic = config_dict["mechanism"]["probabilistic_cfg"]["heuristic"]
        tau = str(config_dict["mechanism"]["probabilistic_cfg"]["tau"])
        external_update_on_cached_results = str(
            config_dict["mechanism"]["probabilistic_cfg"][
                "external_update_on_cached_results"
            ]
        )

        learning_rate = str(
            config_dict["mechanism"]["probabilistic_cfg"]["learning_rate"]
        )

        if config_dict["planner"]["method"] == "NoCuts" and exact_match_caching == True:
            mechanism_type += "+Cache"
        if (
            config_dict["planner"]["method"] == "MinCuts"
            and exact_match_caching == True
        ):
            mechanism_type += "+TreeCache"
        if (
            config_dict["planner"]["method"] == "MaxCuts"
            and exact_match_caching == True
        ):
            mechanism_type += "+PerPartitionCache"
        key += (
            mechanism_type
            + "+"
            + heuristic
            + "+lr"
            + learning_rate
            + "+tau"
            + tau
            + "+cachedupdates"
            + external_update_on_cached_results
        )
        if warmup == "True":
            key += "+warmup"
    return (
        key,
        mechanism_type,
        heuristic,
        warmup,
        learning_rate,
        tau,
        external_update_on_cached_results,
    )


def save_logs(log_dict, save_dir):
    log_path = LOGS_PATH.joinpath(save_dir).joinpath(
        f"{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}/result.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        json_object = json.dumps(log_dict, indent=4)
        fp.write(json_object)


# def save_mlflow_artifacts(log_dict):
#     """
#     Write down some figures directly in Mlflow instead of having to fire Plotly by hand in a notebook
#     See also: `analysis.py`
#     """
#     artifacts_dir = LOGS_PATH.joinpath("mlflow_artifacts")
#     artifacts_dir.mkdir(parents=True, exist_ok=True)
#     plot_budget_utilization_per_block(block_log=log_dict["blocks"]).write_html(
#         artifacts_dir.joinpath("budget_utilization.html")
#     )
#     plot_task_status(task_log=log_dict["tasks"]).write_html(
#         artifacts_dir.joinpath("task_status.html")
#     )

#     mlflow.log_artifacts(artifacts_dir)
