from typing import Any, Dict, List

import ray
from loguru import logger
from ray import tune

from turbo.run_simulation import Simulator
from turbo.utils.utils import LOGS_PATH, RAY_LOGS


def run_and_report(config: dict, replace=False) -> None:
    logs = Simulator(config).run()

    if logs:
        tune.report(**logs)


def grid_online(
    initial_blocks: List[int],
    initial_tasks: List[int],
    max_blocks: List[int],
    logs_dir: str,
    tasks_path: List[str],
    blocks_path: str,
    blocks_metadata: str,
    mechanism: List[str],
    mock: int = True,
    block_requests_pattern: List[int] = [1],
    planner: List[str] = ["MinCuts"],
    avg_num_tasks_per_block: List[int] = [100],
    block_selection_policy: List[str] = ["RandomBlocks"],
    max_tasks: List[int] = [None],
    enable_random_seed: bool = False,
    global_seed: int = 64,
    alpha: List[int] = [0.05],
    beta: List[int] = [0.001],
    learning_rate: List[int] = [0.1],
    heuristic: str = "bin_updates:100:5",
    zipf_k: List[int] = [0.5],
    variance_reduction: List[bool] = False,
    log_every_n_tasks: int = 100,
    bootstrapping: bool = [True],
    exact_match_caching: bool = [True],
    mlflow_random_prefix: bool = False,
    validation_interval: int = 0,
    mlflow_experiment_id: str = "turbo",
    save_logs: bool = False,
    tau: List[int] = [0],
    external_update_on_cached_results: List[bool] = [True],
    enable_mlflow=False,
):

    config = {
        "mock": mock,
        "puredp": True,
        "n_processes": 1,
        "exact_match_caching": tune.grid_search(exact_match_caching),
        "variance_reduction": variance_reduction,
        "alpha": tune.grid_search(alpha),
        "beta": tune.grid_search(beta),
        "global_seed": global_seed,
        "enable_random_seed": enable_random_seed,
        "mechanism": {
            "type": tune.grid_search(mechanism),
            "probabilistic_cfg": {
                "learning_rate": tune.grid_search(learning_rate),
                "heuristic": tune.grid_search(heuristic),
                "bootstrapping": tune.grid_search(bootstrapping),
                "tau": tune.grid_search(tau),
                "external_update_on_cached_results": tune.grid_search(
                    external_update_on_cached_results
                ),
            },
        },
        "planner": {
            "method": tune.grid_search(planner),
            "monte_carlo": False,
            "monte_carlo_N": 10000,
            "monte_carlo_cache": True,
        },
        "budget_accountant": {"epsilon": 10, "delta": 1e-07},
        "blocks": {
            "initial_num": tune.grid_search(initial_blocks),
            "max_num": tune.grid_search(max_blocks),
            "arrival_interval": 1,
            "block_data_path": blocks_path,
            "block_metadata_path": blocks_metadata,
            "block_requests_pattern": tune.grid_search(block_requests_pattern)
            if not isinstance(block_requests_pattern[0], int)
            else block_requests_pattern,
        },
        "tasks": {
            "path": tune.grid_search(tasks_path),
            "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
            "block_selection_policy": tune.grid_search(block_selection_policy),
            "max_num": tune.grid_search(max_tasks),
            "initial_num": tune.grid_search(initial_tasks),
            "zipf_k": tune.grid_search(zipf_k),
        },
        "logs": {
            "verbose": False,
            "save": save_logs,
            "save_dir": "",
            "mlflow": enable_mlflow,
            "mlflow_experiment_id": mlflow_experiment_id,
            "loguru_level": "INFO",
            "log_every_n_tasks": log_every_n_tasks,
            "print_pid": False,
            "mlflow_random_prefix": mlflow_random_prefix,
            "validation_interval": validation_interval,
            "max_validation_tasks": 1000,
        },
    }

    logger.info(f"Tune config: {config}")

    experiment_analysis = tune.run(
        run_and_report,
        config=config,
        # resources_per_trial={"cpu": 1},
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS.joinpath(logs_dir),
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=["n_allocated_tasks", "total_tasks", "global_budget"],
            parameter_columns={
                "exact_match_caching": "exact_match_caching",
                "planner/method": "planner",
                "mechanism/type": "mechanism",
                "tasks/zipf_k": "zipf_k",
                "mechanism/probabilistic_cfg/heuristic": "heuristic",
                "mechanism/probabilistic_cfg/learning_rate": "learning_rate",
                "mechanism/probabilistic_cfg/bootstrapping": "bootstrapping",
            },
            max_report_frequency=60,
        ),
    )
    # all_trial_paths = experiment_analysis._get_trial_paths()
    # experiment_dir = Path(all_trial_paths[0]).parent


class CustomLoggerCallback(tune.logger.LoggerCallback):

    """Custom logger interface"""

    def __init__(self, metrics=[]) -> None:
        self.metrics = ["n_allocated_tasks"]
        self.metrics.extend(metrics)
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info([f"{key}: {result[key]}" for key in self.metrics])
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
